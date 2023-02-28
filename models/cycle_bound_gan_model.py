from typing import Union, Any

import torch
import itertools

from torch import Tensor

from util.image_pool import ImagePool
from models.base_model import BaseModel
from models import networks
import torchvision.transforms as T
import matplotlib.pyplot as plt
from models.matcher import build_matcher, SetCriterion
import numpy as np


class CycleBoundGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    loss_G: Union[Union[Tensor, int], Any]
    loss_cycle_A: Union[Tensor, Any]
    loss_cycle_B: Union[Tensor, Any]
    idt_A: Tensor
    idt_B: Tensor
    loss_D_A: Union[float, Any]
    loss_D_B: Union[float, Any]
    rec_A: Tensor
    rec_B: Tensor
    real_A: Tensor
    real_B: Tensor
    fake_A: Tensor
    fake_B: Tensor

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or
            test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following
        losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)
        (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument(
                '--lambda_identity',
                type=float,
                default=0.5,
                help='use identity mapping. Setting lambda_identity other than 0 has an effect of '
                     'scaling the weight of the identity mapping loss. For example, if the weight of '
                     'the identity loss should be 10 times smaller than the weight of the '
                     'reconstruction loss, please set lambda_identity = 0.1'
            )

        return parser

    def __init__(self, opt, loaded_model=None):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # if not loaded_model:
        #     raise Exception("You need to specify load_detection_model flag to be able to load DETR model")
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if identity loss is used, we also visualize idt_B=G_A(B) and idt_A=G_A(B)
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            use_sigmoid = False
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, self.device,
                                            opt.n_layers_D, opt.norm_discriminator, use_sigmoid, opt.init_type,
                                            opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, self.device,
                                            opt.n_layers_D, opt.norm_discriminator, use_sigmoid, opt.init_type,
                                            opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionBbx = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # load detr
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        matcher = build_matcher()
        self.criterionBbox = SetCriterion(num_classes=2, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=0.1, losses=['boxes'])
        # self.detr = self.load_detr().to(self.device)
        # self.detr = loaded_model.to(self.device)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.bbox_B = input['B_bbox' if AtoB else 'A_bbox']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)


        for box in self.bbox_B:
            box["labels"] = torch.LongTensor([1])
            try:
                box["boxes"] = torch.Tensor(box["bbox"]).to(self.device)
            except:
                continue
        adapted_box = self.adapt_bbox(self.bbox_B)
        if adapted_box:
            lambda_bbox = 0.5
            lambda_giou = 0.5
            for i in range(len(adapted_box)):
                boxes = adapted_box[i]["boxes"]
                boxes[:, 2:] += boxes[:, :2]
                # boxes[:, 0::2].clamp_(min=0, max=512)
                # boxes[:, 1::2].clamp_(min=0, max=512)
                # boxes = self.box_xyxy_to_cxcywh(boxes)
                # boxes = boxes / 512
                adapted_box[i]["boxes"] = boxes
        masks = []
        # for i, boxes in enumerate(adapted_box):
        #     mask = self.create_mask_gaussian(boxes)
        #     masks.append(mask)
        # print("Mask generation finished quickly")
        #
        # # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.custom_l1_loss(masks, self.rec_A, self.real_A) * lambda_A
        # # Backward cycle loss || G_A(G_B(B)) - B||
        # self.loss_cycle_B = self.custom_l1_loss(masks, self.rec_B, self.real_B) * lambda_B
        try:
            for i, boxes in enumerate(adapted_box):
                mask = self.create_mask(self.fake_B, boxes)
                masks.append(mask)
            print("Mask generation finished quickly")

            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.custom_l1_loss(masks, self.rec_A, self.real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.custom_l1_loss(masks, self.rec_B, self.real_B) * lambda_B
        except:
            print("Some error occured")
            self.loss_cycle_A = 0
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = 0
        # if adapted_box:
        #     lambda_bbox = 0.5
        #     lambda_giou = 0.5
        #     for i in range(len(adapted_box)):
        #         boxes = adapted_box[i]["boxes"]
        #         boxes[:, 2:] += boxes[:, :2]
        #         # boxes[:, 0::2].clamp_(min=0, max=512)
        #         # boxes[:, 1::2].clamp_(min=0, max=512)
        #         # boxes = self.box_xyxy_to_cxcywh(boxes)
        #         # boxes = boxes / 512
        #         adapted_box[i]["boxes"] = boxes
            # import pdb;pdb.set_trace()
            # self.loss_bounding_box = lambda_bbox * self.criterionBbox(self.detr(self.fake_B), adapted_box)['loss_bbox']
            # self.loss_giou = lambda_giou * self.criterionBbox(self.detr(self.fake_B), adapted_box)['loss_giou']
            # # combined loss and calculate gradients
            # self.loss_G = self.loss_G_A + self.loss_G_B + \
            #               self.loss_cycle_A + self.loss_cycle_B + \
            #               self.loss_idt_A + self.loss_idt_B + self.loss_bounding_box + self.loss_giou
        # else:
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def adapt_bbox(self, bbox_B):
        if bbox_B:
            batch_samples = []
            for i in range(len(bbox_B[0].get('image_id'))):
                boxes_obj = {'boxes': [], 'labels': [], 'image_id': bbox_B[0].get('image_id')[i]}
                for box in bbox_B:
                    boxes_obj['boxes'].append([x[i] for x in box['bbox']])
                    # TODO: change to more flexible labels for more classes
                    boxes_obj['labels'].append(1)
                boxes_obj['boxes'] = torch.Tensor(boxes_obj['boxes']).to(self.device)
                batch_samples.append(boxes_obj)
            return batch_samples
        else:
            print("@Bbox loss could not be calculated as annotation is missing")
            return None

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no_pre_trans gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    # transform = T.Compose([
    #     T.Resize(800),
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    #
    # COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    #           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    #
    # finetuned_classes = [
    #     'N/A',
    #     'catheter'
    # ]
    # for output bounding box post-processing
    def box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def plot_finetuned_results(self, pil_img, prob=None, boxes=None, name=False):
        plt.figure(figsize=(16, 10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = self.COLORS * 100
        if prob is not None and boxes is not None:
            for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, color=c, linewidth=3))
                cl = p.argmax()
                text = f'{self.finetuned_classes[cl]}: {p[cl]:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        if name:
            plt.savefig(name)
        else:
            plt.show()

    def filter_bboxes_from_outputs(self, outputs, im, threshold=0.7):

        # keep only predictions with confidence above threshold
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        probas_to_keep = probas[keep]

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

        return probas_to_keep, bboxes_scaled

    def run_workflow(self, my_image, my_model, name=False):
        # mean-std normalize the input image (batch-size: 1)
        #     img = my_image.unsqueeze(0)
        #     img = my_image.permute(1,2,0)
        img = self.transform(my_image).unsqueeze(0)
        # propagate through the model
        outputs = my_model(img)

        for threshold in [0.9, 0.01]:

            probas_to_keep, bboxes_scaled = self.filter_bboxes_from_outputs(outputs, my_image, threshold=threshold)
            if not name:
                self.plot_finetuned_results(my_image,
                                            probas_to_keep,
                                            bboxes_scaled)
            else:
                self.plot_finetuned_results(my_image,
                                            probas_to_keep,
                                            bboxes_scaled,
                                            name)

    def custom_l1_loss(self, masks, outputs, targets):
        loss = 0
        for mask, one_output, one_target in zip(masks, outputs, targets):
            masked_output = torch.Tensor(mask).to(self.device) * abs(one_output - one_target)
            loss += torch.mean(masked_output) * (512*512*3)/np.sum(mask)
        return loss

    # def create_mask(self, img, boxes):
    #     m = np.full((img.shape[1], img.shape[2]), 0.2)
    #     for box in boxes:
    #
    #         for i in range(img.shape[1]):
    #             for j in range(img.shape[2]):
    #                 if box[0] < j < box[2] and box[1] < i < box[3]:
    #                     x = np.random.normal(0.8, 0.1)
    #                     if x > 1:
    #                         x = 1
    #                     if x < 0.7:
    #                         x = 0.7
    #                     m[i][j] = x
    #
    #     return np.repeat(np.expand_dims(m, 0), 3, axis=0)

    # def create_mask(self, img, boxes):
    #     m = np.clip(np.random.normal(0.1, 0.05, (img.shape[1], img.shape[2])), 0.05, 0.2)
    #     for box in boxes['boxes']:
    #         m[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = np.clip(np.random.normal(0.8, 0.1, m[int(box[1]):int(box[3]), int(box[0]):int(box[2])].shape), 0.6, 1)
    #     return np.repeat(np.expand_dims(m, 0), 3, axis=0)

    def create_mask(self, img, boxes):
        if len(img.shape) == 4:
            m = np.full((img.shape[2], img.shape[3]), 0.1)
        elif len(img.shape) == 3:
            m = np.full((img.shape[1], img.shape[2]), 0.1)
        for box in boxes['boxes']:
            m[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = np.full(
                m[int(box[1]):int(box[3]), int(box[0]):int(box[2])].shape, 0.8)
        return np.repeat(np.expand_dims(m, 0), 3, axis=0)

    # def create_mask_gaussian(self, boxes):
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from scipy.stats import multivariate_normal
    #
    #     kernels = []
    #
    #     for box in boxes['boxes']:
    #         dist_1 = box[2] - box[0]
    #         dist_2 = box[3] - box[1]
    #         mean_x, mean_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    #         m = (mean_x, mean_y)
    #         s = [[dist_1 * 6, 0], [0, dist_2 * 6]]
    #         k = multivariate_normal(mean=m, cov=s)
    #         kernels.append(k)
    #
    #     # create a grid of (x,y) coordinates at which to evaluate the kernels
    #     xlim = (0, 512)
    #     ylim = (0, 512)
    #     xres = 512
    #     yres = 512
    #
    #     x = np.linspace(xlim[0], xlim[1], xres)
    #     y = np.linspace(ylim[0], ylim[1], yres)
    #     xx, yy = np.meshgrid(x, y)
    #
    #     # evaluate kernels at grid points
    #     xxyy = np.c_[xx.ravel(), yy.ravel()]
    #     zz = kernels[0].pdf(xxyy) + kernels[1].pdf(xxyy) + kernels[2].pdf(xxyy)
    #
    #     # reshape and plot image
    #     img = np.clip(zz.reshape((xres, yres)) / np.max(zz.reshape((xres, yres))), 0.1, None)
    #
    #     return np.repeat(np.expand_dims(img, 0), 3, axis=0)

    # def create_mask(self, img, boxes):
    #     m = np.full((img.shape[1], img.shape[2]), 0.1)
    #     for box in boxes['boxes']:
    #         m[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = np.random.normal(0.8, 0.1, m[int(box[1]):int(box[3]), int(box[0]):int(box[2])].shape)
    #         # for i in range(img.shape[1]):
    #         #     for j in range(img.shape[2]):
    #         #         if box[0] < j < box[2] and box[1] < i < box[3]:
    #         #             m[i][j] = np.random.normal(0.8, 0.1)
    #     return np.repeat(np.expand_dims(m, 0), 3, axis=0)

    # def load_detr(self):
    #     """Load bounding box detection model to obtain bounding box predictions for generated images"""
    #     # state_dict = torch.load(path, map_location=str(self.device))
    #     # model.load_state_dict(state_dict)
    #     # return model
    #     # standard PyTorch mean-std input image normalization
    #
    #     num_classes = 2
    #     model_detr = torch.hub.load('woctezuma/detr',
    #                                 'detr_resnet50',
    #                                 pretrained=False,
    #                                 num_classes=num_classes,
    #                                 )
    #     # model_detr = torch.hub.load('models/detr',
    #     #                             'detr_resnet50',
    #     #                             pretrained=False,
    #     #                             num_classes=num_classes,
    #     #                             source='local')
    #
    #     checkpoint = torch.load('finetuned_detr_2cl_300ep/checkpoint.pth',
    #                             map_location='cpu')
    #
    #     model_detr.load_state_dict(checkpoint['model'],
    #                                strict=False)
    #
    #     model_detr.eval()
    #     return model_detr

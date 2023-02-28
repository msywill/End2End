from typing import Union, Any

import torch
import itertools

from torch import Tensor

from util.image_pool import ImagePool
from models.base_model import BaseModel
from models import networks
from models.parts_model import StyleLoss, PerceptualLoss
from models.bigan_64 import Encoder, Discriminator, FeatureExtractor
import cv2
import numpy as np


class CycleMedGANModel(BaseModel):
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

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'style', 'perceptual']
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
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
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
        try:
            for i, boxes in enumerate(adapted_box):
                mask = self.create_mask(self.fake_B[i], boxes)
                masks.append(mask)
            print("Mask generation finished quickly")

            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.custom_l1_loss(masks, self.rec_A, self.real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.custom_l1_loss(masks, self.rec_B, self.real_B) * lambda_B
        except:
            self.loss_cycle_A = 0
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = 0

        # # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # # Backward cycle loss || G_A(G_B(B)) - B||
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # CYCLE MED GAN LOSSES
        size = (64, 64)

        real_A_resized = torch.Tensor(cv2.resize(np.array(self.real_A[0].permute(1, 2, 0).cpu().detach()),
                                                 dsize=size,
                                                 interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).unsqueeze(0)
        rec_A_resized = torch.Tensor(cv2.resize(np.array(self.rec_A[0].permute(1, 2, 0).cpu().detach()),
                                                dsize=size,
                                                interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).unsqueeze(0)
        real_B_resized = torch.Tensor(cv2.resize(np.array(self.real_B[0].permute(1, 2, 0).cpu().detach()),
                                                 dsize=size,
                                                 interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).unsqueeze(0)
        rec_B_resized = torch.Tensor(cv2.resize(np.array(self.rec_B[0].permute(1, 2, 0).cpu().detach()),
                                                dsize=size,
                                                interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).unsqueeze(0)
        fixed_layers = ['conv1x', 'conv2x', 'conv3x', 'conv4x', 'conv4_1x', 'conv5x', 'conv1z', 'conv2z', 'conv1xz',
                        'conv2xz', 'conv3xz']
        extractor_1, extractor_2, E = self.load_external_bigan_model_parts(fixed_layers)
        feature_map_1 = extractor_1(real_A_resized.to(self.device),
                                    E.forward(torch.randn(real_A_resized.size()).to(self.device)))
        feature_map_2 = extractor_2(rec_A_resized.to(self.device),
                                    E.forward(torch.randn(rec_A_resized.size()).to(self.device)))
        feature_map_3 = extractor_1(real_B_resized.to(self.device),
                                    E.forward(torch.randn(real_A_resized.size()).to(self.device)))
        feature_map_4 = extractor_2(rec_B_resized.to(self.device),
                                    E.forward(torch.randn(rec_A_resized.size()).to(self.device)))

        lambda_p = self.opt.lambda_perceptual
        lambda_s = self.opt.lambda_style
        # Cycle-Perceptual Loss
        self.loss_perceptual = PerceptualLoss(fixed_layers, lambda_p=lambda_p)(feature_map_1, feature_map_2) + \
                               PerceptualLoss(fixed_layers, lambda_p=lambda_p)(feature_map_3, feature_map_4)

        # Style loss
        self.loss_style = StyleLoss(fixed_layers, lambda_s=lambda_s)(feature_map_1, feature_map_2) + \
                          StyleLoss(fixed_layers, lambda_s=lambda_s)(feature_map_3, feature_map_4)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + self.loss_style + self.loss_perceptual
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
        self.backward_D_B()      # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def load_external_bigan_model_parts(self,
                                        lays,
                                        discriminator_path="checkpoints_bigan/bigan_cath_A_64_1000/1000_D.pth",
                                        encoder_path="checkpoints_bigan/bigan_cath_A_64_1000/1000_E.pth",
                                        ):
        # load pretrained discriminator module
        D_1 = Discriminator(256, False).to(self.device)
        D_2 = Discriminator(256, False).to(self.device)

        state_dict_discr = torch.load(discriminator_path, map_location=str(self.device))
        D_1.load_state_dict(state_dict_discr)
        D_2.load_state_dict(state_dict_discr)

        extractor_1 = FeatureExtractor(D_1, lays)
        extractor_2 = FeatureExtractor(D_2, lays)

        # load pretrained encoder module
        E = Encoder(256).to(self.device)
        state_dict_enc = torch.load(encoder_path, map_location=str(self.device))
        E.load_state_dict(state_dict_enc)

        # change mode to eval mode
        D_1.eval()
        D_2.eval()
        E.eval()

        return extractor_1, extractor_2, E

    def custom_l1_loss(self, masks, outputs, targets):
        loss = 0
        for mask, one_output, one_target in zip(masks, outputs, targets):
            masked_output = torch.Tensor(mask).to(self.device) * abs(one_output - one_target)
            loss += torch.mean(masked_output) * (512*512*3)/np.sum(mask)
        return loss

    # def create_mask(self, img, boxes):
    #     m = np.full((img.shape[1], img.shape[2]), 0.1)
    #     for box in boxes['boxes']:
    #         m[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = np.random.normal(0.8, 0.1, m[int(box[1]):int(box[3]), int(box[0]):int(box[2])].shape)
    #         # for i in range(img.shape[1]):
    #         #     for j in range(img.shape[2]):
    #         #         if box[0] < j < box[2] and box[1] < i < box[3]:
    #         #             m[i][j] = np.random.normal(0.8, 0.1)
    #     return np.repeat(np.expand_dims(m, 0), 3, axis=0)

    def create_mask(self, img, boxes):
        m = np.clip(np.random.normal(0.1, 0.05, (img.shape[1], img.shape[2])), 0.05, 0.2)
        for box in boxes['boxes']:
            m[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = np.clip(np.random.normal(0.8, 0.1, m[int(box[1]):int(box[3]), int(box[0]):int(box[2])].shape), 0.6, 1)
        return np.repeat(np.expand_dims(m, 0), 3, axis=0)

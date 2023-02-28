import torch
import itertools
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool
from monai.losses import DiceLoss
from cldice_loss.pytorch import cldice
from monai.transforms import Activations, AsDiscrete, Compose



class CycleGANCldiceUnetModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    # soft_dice: DiceLoss

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
            parser.add_argument('--lambda_2', type=float, default=0.5, help='weight of segmentation loss')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_unet = 0
        if opt.continue_train:
            self.loss_G_A = 0
            self.loss_G_B =0
            self.loss_D_A = 0
            self.loss_D_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            self.loss_cycle_A = 0
            self.loss_cycle_B = 0
#         if self.opt.train_phase == 2 or self.opt.train_phase == 4:
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'unet'] # loss_seg

        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        # visual_names_A = ['real_A', 'fake_B', 'rec_A']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B']
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        # visual_names_U = ['real_mask', 'outputs_binary']
        visual_names_U = ['real_mask']

        # if identity loss is used, we also visualize idt_B=G_A(B) and idt_A=G_A(B)
        # if self.isTrain and self.opt.lambda_identity > 0.0:
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B + visual_names_U
        # specify the models you want to save to the disk. The training/test scripts will call
        # 选择需要存储的网络的名字
        # <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'u_net']
            # self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # load pretrained unet
        self.u_net = torch.load(self.opt.load_unet, map_location='cpu').to(self.device)

        # create soft_dice_cldice object
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.Soft_Cl_dice = cldice.soft_dice_cldice()
        self.soft_dice = DiceLoss(sigmoid=True, smooth_nr=1.0, smooth_dr=1.0)

        # create optimizer for unet
        self.optimizer_U = torch.optim.Adam(self.u_net.parameters(), lr=opt.lr)

        if self.isTrain:  # define discriminators
            use_sigmoid = False
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, self.device,
                                            opt.n_layers_D, opt.norm_discriminator, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, self.device,
                                            opt.n_layers_D, opt.norm_discriminator, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)


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
            self.optimizers.append(self.optimizer_U)


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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # self.mask_paths = input['maskA_paths']
        self.real_mask = input['maskA'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))



    def forward_gan_unet(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)  # 在这一行就不行了
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        self.fake_low = self.u_net(self.fake_B)



    def forward_unet(self):
        self.fake_B = self.netG_A(self.real_A)
        self.fake_A = self.netG_B(self.real_B)
        fake_B_u = self.fake_B.detach()
        # self.fake_low = self.fake_B_pool.query(self.fake_B)
        self.fake_mask = self.u_net(fake_B_u) # 不影响GA的参数


    def backward_unet(self):
        loss_soft_dice = self.soft_dice(self.fake_mask, self.real_mask)
        outputs_binary = self.post_trans(self.fake_mask)
        self.loss_unet = self.Soft_Cl_dice.forward(self.real_mask, outputs_binary, loss_soft_dice)
        self.loss_unet.backward()


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
        pred_fake = netD(fake.detach())  #为了不影响generator的参数
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)  # 如果batch==1 fakeb只有一张 是不是可以不需要pool
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
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def backward_G_and_U(self):
        """Calculate the loss for generators G_A and G_B and Unet"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_2 = self.opt.lambda_2
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
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # cl dice loss for segmentation
        loss_soft_dice = self.soft_dice(self.fake_low, self.real_mask)
        self.outputs_binary = self.post_trans(self.fake_low)
        self.loss_seg = self.Soft_Cl_dice.forward(self.real_mask, self.outputs_binary, loss_soft_dice) * lambda_2


        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + self.loss_seg

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        if self.opt.train_phase == 1:
            # forward
            self.forward()  # G_A 和 G_B 参与
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # 因为在上一轮更新结束时 D_A 和 D_B gradient都是True
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()  # calculate gradients for G_A and G_B
            self.optimizer_G.step()  # update G_A and G_B's weights
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights # 结束的时候 G 和 D都 requires = true

        elif self.opt.train_phase == 2:
            self.forward_unet() # 得到fake image 和 prediction
            # self.set_requires_grad([self.netD_A, self.netD_B], False)
            # self.set_requires_grad([self.netG_A, self.netG_B], False) # 冻结 cycleGAN
            # self.set_requires_grad([self.u_net], True)
            self.optimizer_U.zero_grad() # 更新unet
            self.backward_unet()
            self.optimizer_U.step() # 只有传入optimizer的参数会被更新 结束的时候 unet requires_grad = true

        elif self.opt.train_phase == 3:
            # forward
            self.forward_gan_unet()  # G_A G_B UNet都参与 此时unet requires_grad = true
            self.set_requires_grad([self.u_net], False)  # freeze unet
            # self.set_requires_grad([self.netG_A, self.netG_B], True)
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G_and_U()  # calculate gradients for G_A and G_B with additional segmentation loss
            self.optimizer_G.step()  # update G_A and G_B's weights
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights  # 结束时unet = false

        elif self.opt.train_phase == 4:
            self.set_requires_grad([self.u_net], True)
            self.forward_unet() # 得到fake image 和 prediction
            # self.set_requires_grad([self.netD_A, self.netD_B], False)
            # self.set_requires_grad([self.netG_A, self.netG_B], False) # 冻结 cycleGAN
            # self.set_requires_grad([self.u_net], True)
            self.optimizer_U.zero_grad() # 更新unet
            self.backward_unet()
            self.optimizer_U.step() # 只有传入optimizer的参数会被更新 结束的时候 unet requires_grad = true


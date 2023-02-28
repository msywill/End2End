import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from torchvision.transforms import ToTensor
import random
from typing import List, Any


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_maskA = os.path.join(opt.dataroot, opt.maskA) # /path/to/data/maskA'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size)) # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.maskA_paths = sorted(make_dataset(self.dir_maskA, opt.max_dataset_size)) ## 这里看一下make_dataset方法
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.maskA_size = len(self.maskA_paths) # TODO: assert A_size != maskA_size
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1)) # inputchannel为1时转换为grayscale
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        # self.transform_mask = get_transform(self.opt)
        self.transform_mask = ToTensor()


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            maskA(tensor)    -- its corresponding mask
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            maskA_paths (str)-- maskA paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        maskA_path = self.maskA_paths[index % self.maskA_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        maskA_img  = Image.open(maskA_path)    # 本身就是 grayscale

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        maskA = self.transform_mask(maskA_img)  # 这里load之后应该直接是（1，512，512）
        # transform_mask = ToTensor()
        # maskA = transform_mask(maskA_img)

        return {'A': A, 'B': B, 'maskA': maskA, 'A_paths': A_path, 'B_paths': B_path, 'maskA_paths': maskA_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


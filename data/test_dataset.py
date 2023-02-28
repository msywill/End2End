from . import unaligned_dataset
import sys
sys.path.append("..")
from data import create_dataset
from options.base_options import BaseOptions

if __name__ == '__main__':
    opt = BaseOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
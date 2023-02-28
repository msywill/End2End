import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from evaluate import metrics
from models import create_model
from data import create_dataset

from options.eval_options import EvalOptions
from train_cyclegan import load_detr



dict_path = '/catheter-CycleGAN/checkpoints_cyclegan/old_cycle_unet/cycle_unet_parallel/80_u_net.pth'
model_path = '/Users/mengsiyue/PycharmProjects/remote/pretrained-Unet/best_metric_model_ep41_0-922.pth'

# model = torch.load(model_path)
# model.load_state_dict(torch.load(dict_path))

state_dict = torch.load(dict_path)
if hasattr(state_dict, '_metadata'):
    del state_dict._metadata

# patch InstanceNorm checkpoints prior to 0.4
for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
net.load_state_dict(state_dict)
# if __name__ == "__main__":
#     opt = EvalOptions().parse()
#
#     dataset = create_dataset(opt)
#     dataset_size = len(dataset)  # get the number of images in the dataset.
#     print('The number of training images = %d' % dataset_size)
#     if opt.load_detection_model:
#         # load DETR on start
#         model_detr = load_detr()
#         print('Detr loaded')
#         model = create_model(opt, model_detr)  # create a model given opt.model and other options
#     else:
#         model = create_model(opt)
#
#     if opt.metric == 'inception':
#         metrics = [metrics.InceptionScore()]
#     elif opt.metric == 'fcn':
#         metrics = [metrics.FCNScore()]
#     elif opt.metric == 'all':
#         metrics = [metrics.InceptionScore(), metrics.FCNScore()]
#
#     for metric in metrics:
#         metric_scores = np.zeros(len(dataset))
#         for i, data in enumerate(dataset):
#             metric_scores[i] = 0

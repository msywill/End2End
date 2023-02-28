import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import soft_skel
from torchvision.transforms import ToPILImage

class soft_cldice(nn.Module):
    def __init__(self, iter_=30, smooth = 1.0):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    #def forward(y_true, y_pred):
    def forward(self, y_true, y_pred):
        '''

        :param y_true:  tensor of shape(N, C, H, W)
        :param y_pred:  tensor of shape(N, C, H, W)
        :return: loss: tensor of shape(1,)
        '''
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.mul(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.mul(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)

        # tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        # tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)

        cl_dice = 1.0 - 2.0*(tprec*tsens)/(tprec+tsens)
        # 1-cldice
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1.0
    intersection = torch.sum((y_true * y_pred))
    coeff = (2.0 *  intersection + smooth) / (torch.sum(y_true + y_pred) + smooth)
    # probs = torch.sigmoid(y_pred)
    # intersection = torch.sum(y_true * probs)
    # ground_o = torch.sum(y_true)
    # pred_o = torch.sum(probs)
    # denominator = ground_o + pred_o
    # f: torch.Tensor = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    # coeff = (2. *  intersection + float(smooth)) / (torch.sum(y_true + probs) + float(smooth))
    return (1. - coeff)
    # return f


#class soft_dice_cldice(nn.Module):
class soft_dice_cldice(soft_cldice):
    def __init__(self, iter_= 25, alpha=0.1, smooth = 1.0):
        super().__init__()
        # super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_true, y_pred, dice):
        # dice = soft_dice(y_true, y_pred)
        # show = ToPILImage()
        # show(skel_true.squeeze(dim=0)).show()
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.mul(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.mul(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec*tsens) / (tprec+tsens)
        return (1.0 - self.alpha) * dice + self.alpha*cl_dice

import torch
from torch import nn


class GramMatrix(nn.Module):
    def forward(self, input):
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        a, b, c, d = input.size()

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        # in the paper hi, wi, di
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    '''
    Cycle-Style Loss defined in the paper https://arxiv.org/pdf/1903.03374.pdf.
    '''

    def __init__(self, lays, lambda_s):
        super(StyleLoss, self).__init__()
        self.lays = lays
        self.gram = GramMatrix()
        self.lambda_s = lambda_s

    def forward(self, x_model1, x_model2):
        loss = 0
        for lay in self.lays:
            gram1 = self.gram.forward(x_model1[lay])
            gram2 = self.gram.forward(x_model2[lay])
            # Frobenius norm of the difference of gram matrices
            try:
                loss += (self.lambda_s / (4 * (x_model1[lay].size()[1])**2)) * torch.norm(gram1 - gram2)
            except:
                import pdb;pdb.set_trace()
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, lays, lambda_p):
        super(PerceptualLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.lays = lays
        self.lambda_p = lambda_p

    def forward(self, fmap_x, fmap_x_rec):
        loss = 0
        for lay in self.lays:
            f_map_x_layer = fmap_x[lay]
            f_map_x_rec_layer = fmap_x_rec[lay]
            loss += self.lambda_p * self.l1(f_map_x_layer, f_map_x_rec_layer)
        return loss
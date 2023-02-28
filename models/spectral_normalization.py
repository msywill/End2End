import torch
from torch.nn import Parameter
from torch import nn
"""
FROM  : https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
"""

# TODO: Spectral normalization does not work in data parallel regime


def l2normalize(vector, eps=1e-15):
    return vector/(vector.norm()+eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, device, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.device = device
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        u.to(self.device)
        v.to(self.device)
        w.to(self.device)

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        n = (w / sigma.expand_as(w)).to(self.device)
        setattr(self.module, self.name, n)

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class NLayerDiscriminatorSN(nn.Module):
    def __init__(self, input_nc, device, ndf=64, n_layers=3, use_sigmoid=False):
        super(NLayerDiscriminatorSN, self).__init__()
        use_bias = False
        self.device = device

        kw = 4
        padw = 1
        sequence = [
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), device),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=kw, stride=2, padding=padw, bias=use_bias), device),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=1, padding=padw, bias=use_bias), device),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), device)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.model.to(device)

    def forward(self, input):
        return self.model(input)

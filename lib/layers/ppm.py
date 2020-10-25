import torch
from torch import nn
from torch.nn import functional as F

from lib.layers.wrappers import make_act, make_conv, make_norm


class PPM(nn.Module):

    def __init__(self, dim_in, ppm_dim=512, pool_scales=(1, 2, 3, 6), **kwargs):
        """

        :param dim_in: (int) input tensor channel
        :param ppm_dim: (int) dilated convolution channel
        :param pool_scales: (tuple) pool_scales of average pooling
        """
        super(PPM, self).__init__()
        conv = kwargs.pop("conv", "Conv2d")
        norm = kwargs.pop("norm", "BN")
        act = kwargs.pop("act", "ReLU")

        self.dim_in = dim_in
        self.ppm_dim = ppm_dim
        self.pool_scales = pool_scales
        self.conv = conv
        self.norm = norm
        self.act = act

        self._mask_layer()

        self.dim_out = ppm_dim * len(pool_scales) + self.dim_in

    def _mask_layer(self):
        ppm = []
        for scale in self.pool_scales:
            ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                make_conv(self.dim_in, self.ppm_dim, kernel_size=1,
                          norm=make_norm(self.ppm_dim, norm=self.norm), act=make_act())
            ))
        self.ppm = nn.ModuleList(ppm)

    def forward(self, x):
        input_size = x.shape
        out = [x]
        for i in range(len(self.ppm)):
            out.append(nn.functional.interpolate(
                self.ppm[i](x), (input_size[2], input_size[3]), mode='bilinear', align_corners=True))

        out = torch.cat(out, 1)
        return out

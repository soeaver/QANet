import torch
from torch import nn
from torch.nn import functional as F

from lib.layers.wrappers import make_act, make_conv, make_norm


class ASPP(nn.Module):

    def __init__(self, dim_in, aspp_dim=256, dilations=(6, 12, 18,), **kwargs):
        """

        :param dim_in: (int) input tensor channel
        :param aspp_dim: (int) dilated convolution channel
        :param dilations: (tuple) dilations of dilated convolution
        """
        super(ASPP, self).__init__()
        conv = kwargs.pop("conv", "Conv2d")
        norm = kwargs.pop("norm", "BN")
        act = kwargs.pop("act", "ReLU")

        self.dim_in = dim_in
        self.aspp_dim = aspp_dim
        self.dilations = dilations
        self.conv = conv
        self.norm = norm
        self.act = act

        self._mask_layer()

        self.dim_out = (len(dilations) + 2) * aspp_dim

    def _mask_layer(self):
        aspp = [make_conv(self.dim_in, self.aspp_dim, kernel_size=1, norm=make_norm(self.aspp_dim, norm=self.norm),
                          act=make_act(act=self.act))]
        for dilation in self.dilations:
            aspp.append(
                make_conv(self.dim_in, self.aspp_dim, kernel_size=3, dilation=dilation, conv=self.conv,
                          norm=make_norm(self.aspp_dim, norm=self.norm), act=make_act(act=self.act))
            )
        self.aspp = nn.ModuleList(aspp)

        self.im_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            make_conv(self.dim_in, self.aspp_dim, kernel_size=1, norm=make_norm(self.aspp_dim, norm=self.norm),
                      act=make_act(act=self.act))
        )

    def forward(self, x):
        input_size = x.size()
        out = [F.interpolate(self.im_pool(x), (input_size[2], input_size[3]), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            out.append(self.aspp[i](x))
        out = torch.cat(out, 1)
        return out

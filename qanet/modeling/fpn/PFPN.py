import torch
import torch.nn as nn

from lib.layers import make_act, make_conv, make_norm

from qanet.modeling import registry
from qanet.modeling.fpn import FPN


@registry.FPN_BODY.register("pfpn")
class PFPN(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        panoptic_dim = cfg.FPN.PANOPTIC.CONV_DIM
        norm = cfg.FPN.PANOPTIC.NORM
        self.spatial_in = spatial_in
        self.use_fpn = cfg.FPN.PANOPTIC.USE_FPN

        if self.use_fpn:
            self.fpn = FPN(cfg, dim_in, self.spatial_in)

        if self.use_fpn:
            self.dim_in = self.fpn.dim_out
        else:
            self.dim_in = dim_in

        self.scale1_block = panoptic_upsampler_block(self.dim_in[3], panoptic_dim, 3, norm=norm)  # 1/32
        self.scale2_block = panoptic_upsampler_block(self.dim_in[2], panoptic_dim, 2, norm=norm)  # 1/16
        self.scale3_block = panoptic_upsampler_block(self.dim_in[1], panoptic_dim, 1, norm=norm)  # 1/8
        self.scale4_block = panoptic_upsampler_block(self.dim_in[0], panoptic_dim, 0, norm=norm)  # 1/4

        self.dim_out = [panoptic_dim]
        if self.use_fpn:
            self.spatial_out = self.fpn.spatial_out[:1]
        else:
            self.spatial_out = spatial_in[:1]

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps gen post FPN, (N, C, H, W)
        Returns:
            segmentation_mask: semantic segmentation mask
        """
        if self.use_fpn:
            x = self.fpn(x)
        x1 = self.scale1_block(x[3])
        x2 = self.scale2_block(x[2])
        x3 = self.scale3_block(x[1])
        x4 = self.scale4_block(x[0])

        x = x1 + x2 + x3 + x4

        return [x]


def panoptic_upsampler_block(dim_in, dim_out, expansion, norm=''):
    modules = []
    if expansion == 0:
        modules.append(make_conv(
            dim_in,
            dim_out,
            kernel=3,
            dilation=1,
            stride=1,
            norm=make_norm(dim_out, norm=norm),
            act=make_act(),
        ))  # no upsample

    for i in range(expansion):
        modules.append(make_conv(
            dim_in if i == 0 else dim_out,
            dim_out,
            kernel=3,
            dilation=1,
            stride=1,
            norm=make_norm(dim_out, norm=norm),
            act=make_act(),
        ))
        modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    return nn.Sequential(*modules)

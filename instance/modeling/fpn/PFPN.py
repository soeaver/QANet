import torch
import torch.nn as nn

from pet.utils.net import make_conv
from pet.instance.modeling.fpn import fpn, nasfpn
from pet.instance.core.config import cfg
from pet.instance.modeling import registry


@registry.FPN_BODY.register("pfpn")
class pfpn(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        assert not (cfg.FPN.PANOPTIC.USE_FPN and cfg.FPN.PANOPTIC.USE_NASFPN), \
            'USE_FPN and USE_NASFPN cannot be true at same time'
        if cfg.FPN.PANOPTIC.USE_FPN:
            self.fpn = fpn(dim_in, spatial_scale)
        if cfg.FPN.PANOPTIC.USE_NASFPN:
            self.fpn = nasfpn(dim_in, spatial_scale)
            
        panoptic_dim = cfg.FPN.PANOPTIC.CONV_DIM
        use_bn = cfg.FPN.PANOPTIC.USE_BN
        use_gn = cfg.FPN.PANOPTIC.USE_GN
        assert not (use_bn and use_gn), 'USE_BN and USE_GN cannot be true at same time'

        if cfg.FPN.PANOPTIC.USE_FPN or cfg.FPN.PANOPTIC.USE_NASFPN:
            self.dim_in = self.fpn.dim_out
        else:
            self.dim_in = dim_in

        self.scale1_block = panoptic_upsampler_block(self.dim_in[3], panoptic_dim, 3, use_bn, use_gn)  # 1/32
        self.scale2_block = panoptic_upsampler_block(self.dim_in[2], panoptic_dim, 2, use_bn, use_gn)  # 1/16
        self.scale3_block = panoptic_upsampler_block(self.dim_in[1], panoptic_dim, 1, use_bn, use_gn)  # 1/8
        self.scale4_block = panoptic_upsampler_block(self.dim_in[0], panoptic_dim, 0, use_bn, use_gn)  # 1/4

        self.dim_out = [panoptic_dim]
        if cfg.FPN.PANOPTIC.USE_FPN or cfg.FPN.PANOPTIC.USE_NASFPN:
            self.spatial_scale = self.fpn.spatial_scale[:1]
        else:
            self.spatial_scale = spatial_scale[:1]

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps gen post FPN, (N, C, H, W)
        Returns:
            segmentation_mask: semantic segmentation mask
        """
        if cfg.FPN.PANOPTIC.USE_FPN or cfg.FPN.PANOPTIC.USE_NASFPN:
            x = self.fpn(x)
        x1 = self.scale1_block(x[3])
        x2 = self.scale2_block(x[2])
        x3 = self.scale3_block(x[1])
        x4 = self.scale4_block(x[0])

        x = x1 + x2 + x3 + x4

        return [x]


def panoptic_upsampler_block(dim_in, dim_out, expansion, use_bn, use_gn):
    modules = []
    if expansion == 0:
        modules.append(make_conv(
            dim_in,
            dim_out,
            kernel=3,
            dilation=1,
            stride=1,
            use_bn=use_bn,
            use_gn=use_gn,
            use_relu=True,
            kaiming_init=True
        ))  # no upsample

    for i in range(expansion):
        modules.append(make_conv(
            dim_in if i == 0 else dim_out,
            dim_out,
            kernel=3,
            dilation=1,
            stride=1,
            use_bn=use_bn,
            use_gn=use_gn,
            use_relu=True,
            kaiming_init=True
        ))
        modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    return nn.Sequential(*modules)

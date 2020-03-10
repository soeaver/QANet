import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pet.utils.net import make_conv
from pet.instance.core.config import cfg
from pet.instance.modeling.fpn import get_min_max_levels
from pet.instance.modeling import registry


class MergingCell(nn.Module):
    def __init__(self, dim_in=256, with_conv=True, use_lite=False, use_bn=False, use_gn=False):
        super(MergingCell, self).__init__()
        self.dim_in = dim_in
        self.with_conv = with_conv
        if self.with_conv:
            self.conv_out = nn.Sequential(
                nn.ReLU(inplace=True),
                make_conv(self.dim_in, self.dim_in, kernel=3, use_dwconv=use_lite, use_bn=use_bn, use_gn=use_gn,
                          use_relu=False, suffix_1x1=use_lite)
            )
        self.dim_out = self.dim_in

    def _binary_op(self, x1, x2):
        raise NotImplementedError

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode='nearest')
        else:
            assert x.shape[-2] % size[-2] == 0 and x.shape[-1] % size[-1] == 0
            kernel_size = (math.ceil(x.shape[-2] / size[-2]), math.ceil(x.shape[-1] / size[-1]))
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size, ceil_mode=True)
            return x

    def forward(self, x1, x2, out_size):
        assert x1.shape[:2] == x2.shape[:2]
        assert len(out_size) == 2

        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)

        x = self._binary_op(x1, x2)
        if self.with_conv:
            x = self.conv_out(x)
        return x


class SumCell(MergingCell):
    def _binary_op(self, x1, x2):
        return x1 + x2


class GPCell(MergingCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

    def _binary_op(self, x1, x2):
        x1_att = self.global_pool(x1).sigmoid()
        return x1 + x2 * x1_att.expand_as(x2)


# ---------------------------------------------------------------------------- #
# Functions for bolting NASFPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
@registry.FPN_BODY.register("nasfpn")
class nasfpn(nn.Module):
    # dim_in = [256, 512, 1024, 2048]
    # spatial_scale = [1/4, 1/8, 1/16, 1/32]
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]  # 2048
        self.spatial_scale = spatial_scale

        self.num_stack = cfg.FPN.NASFPN.NUM_STACK  # 7
        nasfpn_dim = cfg.FPN.NASFPN.DIM  # 256
        use_lite = cfg.FPN.NASFPN.USE_LITE
        use_bn = cfg.FPN.NASFPN.USE_BN
        use_gn = cfg.FPN.NASFPN.USE_GN
        min_level, max_level = get_min_max_levels()  # 3, 7
        self.num_backbone_stages = len(dim_in) - (
                min_level - cfg.FPN.LOWEST_BACKBONE_LVL)  # 3 (cfg.FPN.LOWEST_BACKBONE_LVL=2)

        # nasfpn module
        self.nasfpn_in = nn.ModuleList()
        for i in range(self.num_backbone_stages):
            px_in = make_conv(dim_in[-1 - i], nasfpn_dim, kernel=1, use_bn=use_bn, use_gn=use_gn)
            self.nasfpn_in.append(px_in)
        self.dim_in = nasfpn_dim

        # add nasfpn connections
        self.nasfpn_stages = nn.ModuleList()
        for _ in range(self.num_stack):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GPCell(nasfpn_dim, use_lite=use_lite, use_bn=use_bn, use_gn=use_gn)
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(nasfpn_dim, use_lite=use_lite, use_bn=use_bn, use_gn=use_gn)
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(nasfpn_dim, use_lite=use_lite, use_bn=use_bn, use_gn=use_gn)
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_43_4'] = SumCell(nasfpn_dim, use_lite=use_lite, use_bn=use_bn, use_gn=use_gn)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GPCell(with_conv=False)
            stage['sum_55_5'] = SumCell(nasfpn_dim, use_lite=use_lite, use_bn=use_bn, use_gn=use_gn)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GPCell(with_conv=False)
            stage['sum_77_7'] = SumCell(nasfpn_dim, use_lite=use_lite, use_bn=use_bn, use_gn=use_gn)
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GPCell(nasfpn_dim, use_lite=use_lite, use_bn=use_bn, use_gn=use_gn)
            self.nasfpn_stages.append(stage)

        self.extra_levels = max_level - cfg.FPN.HIGHEST_BACKBONE_LVL  # 2
        for _ in range(self.extra_levels):
            self.spatial_scale.append(self.spatial_scale[-1] * 0.5)

        self.spatial_scale = self.spatial_scale[min_level - 2:]
        self.dim_out = [self.dim_in for _ in range(max_level - min_level + 1)]

        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        px_outs = []
        for i in range(self.num_backbone_stages):  # [P5 - P3]
            px = self.nasfpn_in[i](x[-i - 1])
            px_outs.append(px)

        for _ in range(self.extra_levels):  # P6, P7
            px_outs.insert(0, F.max_pool2d(px_outs[0], 2, stride=2))

        p7, p6, p5, p4, p3 = px_outs  # or: [P6 - P2]
        for stage in self.nasfpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_43_4'](p4_2, p3, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])

        return [p3, p4, p5, p6, p7]  # [P3 - P7]
 

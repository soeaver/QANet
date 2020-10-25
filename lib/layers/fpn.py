import torch.nn as nn
import torch.nn.functional as F

from lib.layers.wrappers import make_act, make_conv, make_norm


class FPN(nn.Module):
    def __init__(self, **kwargs):
        super(FPN, self).__init__()
        dim_in = kwargs.pop("dim_in", [256, 512, 1024, 2048])
        spatial_scale = kwargs.pop("spatial_scale", [1/4, 1/8, 1/16, 1/32])
        keep_backbone = kwargs.pop("keep_backbone", False)
        fpn_dim = kwargs.pop("fpn_dim", 256)
        use_c5 = kwargs.pop("use_c5", True)
        m_only = kwargs.pop("m_only", False)
        norm = kwargs.pop("norm", "")
        min_level = kwargs.pop("min_level", 2)
        max_level = kwargs.pop("max_level", 6)
        lowest_bk_lvl = kwargs.pop("lowest_bk_lvl", 2)
        highest_bk_lvl = kwargs.pop("highest_bk_lvl", 5)
        extra_conv = kwargs.pop("extra_conv", False)

        self.dim_in = dim_in[-1]  # 2048
        self.spatial_scale = spatial_scale
        self.keep_backbone = keep_backbone
        self.use_c5 = use_c5
        self.m_only = m_only
        self.max_level = max_level
        self.lowest_bk_lvl = lowest_bk_lvl
        self.highest_bk_lvl = highest_bk_lvl
        self.extra_conv = extra_conv
        self.num_backbone_stages = len(dim_in) - (min_level - lowest_bk_lvl)  # 4

        output_levels = highest_bk_lvl - lowest_bk_lvl + 1
        self.spatial_scale = self.spatial_scale[:output_levels]
        self.dim_out = [self.dim_in for _ in range(output_levels)]

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

    def _make_layer(self, dim_in, fpn_dim, norm, act=""):
        # P5 in
        self.p5_in = make_conv(self.dim_in, fpn_dim, kernel_size=1, norm=make_norm(fpn_dim, norm=norm),
                               act=make_act(act=act))

        # P5 out
        self.p5_out = make_conv(fpn_dim, fpn_dim, kernel_size=3, norm=make_norm(fpn_dim, norm=norm),
                                act=make_act(act=act))

        # fpn module
        self.fpn_in = []
        self.fpn_out = []
        for i in range(self.num_backbone_stages - 1):  # skip the top layer
            px_in = make_conv(dim_in[-i - 2], fpn_dim, kernel_size=1, norm=make_norm(fpn_dim, norm=norm),
                              act=make_act(act=act))  # P4 to P2
            px_out = make_conv(fpn_dim, fpn_dim, kernel_size=3, norm=make_norm(fpn_dim, norm=norm),
                               act=make_act(act=act))
            self.fpn_in.append(px_in)
            self.fpn_out.append(px_out)
        self.fpn_in = nn.ModuleList(self.fpn_in)  # [P4, P3, P2]
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.dim_in = fpn_dim

        if not self.extra_conv and self.max_level == self.highest_bk_lvl + 1:
            self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
            self.spatial_scale.append(self.spatial_scale[-1] * 0.5)

        if self.extra_conv and self.max_level > self.highest_bk_lvl:
            self.extra_pyramid_modules = nn.ModuleList()
            if self.use_c5:
                self.dim_in = dim_in[-1]
            for i in range(self.highest_bk_lvl + 1, self.max_level + 1):
                self.extra_pyramid_modules.append(
                    make_conv(self.dim_in, fpn_dim, kernel_size=3, stride=2, norm=make_norm(fpn_dim, norm=norm),
                              act=make_act(act=act))
                )
                self.dim_in = fpn_dim
                self.spatial_scale.append(self.spatial_scale[-1] * 0.5)

    def forward(self, x):
        c5_out = x[-1]
        px = self.p5_in(c5_out)
        fpn_output_blobs = [self.p5_out(px)] if not self.m_only else [px]  # [P5] or [M5]
        for i in range(self.num_backbone_stages - 1):  # [P5 - P2]
            cx_out = x[-i - 2]  # C4, C3, C2
            cx_out = self.fpn_in[i](cx_out)  # lateral branch
            if cx_out.size()[2:] != px.size()[2:]:
                px = F.interpolate(px, scale_factor=2, mode='nearest')
            px = cx_out + px
            if self.m_only:
                fpn_output_blobs.insert(0, px)  # [M2 - M5]
            else:
                fpn_output_blobs.insert(0, self.fpn_out[i](px))  # [P2 - P5]

        if hasattr(self, 'maxpool_p6'):
            fpn_output_blobs.append(self.maxpool_p6(fpn_output_blobs[-1]))  # [P2 - P6]

        if hasattr(self, 'extra_pyramid_modules'):
            if self.use_c5:
                p6_in = c5_out
            else:
                p6_in = fpn_output_blobs[-1]
            fpn_output_blobs.append(self.extra_pyramid_modules[0](p6_in))
            for module in self.extra_pyramid_modules[1:]:
                fpn_output_blobs.append(module(F.relu(fpn_output_blobs[-1])))  # [P2 - P6, P7]

        if self.keep_backbone:
            fpn_output_blobs.append(x)

        # use all levels
        return fpn_output_blobs  # [P2 - P6]

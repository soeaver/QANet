import math

import torch.nn as nn

import lib.backbone.hrnet as hr
from lib.layers import make_norm, BasicBlock, Bottleneck

from qanet.modeling import registry


class HRNet(hr.HRNet):
    def __init__(self, cfg, stride=32):
        """ Constructor
        """
        super(HRNet, self).__init__()
        self.dim_in = 3
        self.spatial_in = [1]

        block_1 = Bottleneck
        block_2 = BasicBlock

        base_width = cfg.BACKBONE.HRNET.WIDTH
        use_global = cfg.BACKBONE.HRNET.USE_GLOBAL
        stage_with_conv = cfg.BACKBONE.HRNET.STAGE_WITH_CONV
        norm = cfg.BACKBONE.HRNET.NORM
        stage_with_ctx = cfg.BACKBONE.HRNET.STAGE_WITH_CTX

        self.avg_down = cfg.BACKBONE.HRNET.AVG_DOWN
        self.base_width = base_width
        self.norm = norm
        self.stride = stride

        multi_out = 1 if self.stride == 4 else 4

        self.inplanes = 64  # default 64
        self.conv1 = nn.Conv2d(self.dim_in, 64, 3, 2, 1, bias=False)
        self.bn1 = make_norm(64, norm=norm.replace('Mix', ''))
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = make_norm(64, norm=norm.replace('Mix', ''))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_1, 64, 4, 1, conv=stage_with_conv[0], ctx=stage_with_ctx[0])  # 4 blocks
        self.transition1 = self._make_transition(index=1, stride=2)  # Fusion layer 1: create full and 1/2 resolution

        self.stage2 = nn.Sequential(
            hr.StageModule(block_2, base_width, 2, 2, stage_with_conv[1], norm, stage_with_ctx[1], False),
        )  # Stage 2 with 1 group of block modules, which has 2 branches
        self.transition2 = self._make_transition(index=2, stride=2)  # Fusion layer 2: create 1/4 resolution

        self.stage3 = nn.Sequential(
            hr.StageModule(block_2, base_width, 3, 3, stage_with_conv[2], norm, stage_with_ctx[2], use_global),
            hr.StageModule(block_2, base_width, 3, 3, stage_with_conv[2], norm, stage_with_ctx[2], use_global),
            hr.StageModule(block_2, base_width, 3, 3, stage_with_conv[2], norm, stage_with_ctx[2], use_global),
            hr.StageModule(block_2, base_width, 3, 3, stage_with_conv[2], norm, stage_with_ctx[2], use_global),
        )  # Stage 3 with 4 groups of block modules, which has 3 branches
        self.transition3 = self._make_transition(index=3, stride=2)  # Fusion layer 3: create 1/8 resolution

        self.stage4 = nn.Sequential(
            hr.StageModule(block_2, base_width, 4, 4, stage_with_conv[3], norm, stage_with_ctx[3], use_global),
            hr.StageModule(block_2, base_width, 4, 4, stage_with_conv[3], norm, stage_with_ctx[3], use_global),
            hr.StageModule(block_2, base_width, 4, multi_out, stage_with_conv[3], norm, stage_with_ctx[3], use_global),
        )  # Stage 4 with 3 groups of block modules, which has 4 branches

        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]
        self.spatial_out = self.stage_out_spatial[1:int(math.log(self.stride, 2))]

        del self.incre_modules
        del self.downsamp_modules
        del self.final_layer
        del self.avgpool
        del self.classifier
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        if self.stride == 4:
            return x[:1]
        return x


# ---------------------------------------------------------------------------- #
# HRNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("hrnet_s4")
def hrnet_s4(cfg):
    model = HRNet(cfg, stride=4)
    return model


@registry.BACKBONES.register("hrnet")
def hrnet(cfg):
    model = HRNet(cfg, stride=32)
    return model

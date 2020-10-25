import math
import torch.nn as nn
import torch.nn.functional as F

import lib.backbone.mobilenet_v3 as mv3
from lib.layers import InvertedResidual, make_act, make_norm
from lib.utils.net import convert_conv2convsamepadding_model, make_divisible

from instance.modeling import registry


class MobileNetV3(mv3.MobileNetV3):
    def __init__(self, cfg, act="H_Swish", stride=32):
        """ Constructor
        """
        super(MobileNetV3, self).__init__()
        self.dim_in = 3
        self.spatial_in = [1]

        block = InvertedResidual
        setting = cfg.BACKBONE.MV3.SETTING
        widen_factor = cfg.BACKBONE.MV3.WIDEN_FACTOR
        layers_cfg = mv3.MV3_CFG[setting]
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        channels = [make_divisible(ch * widen_factor, 8) for ch in num_of_channels]
        norm = cfg.BACKBONE.MV3.NORM

        self.widen_factor = widen_factor
        self.channels = channels
        self.norm = norm
        self.act = act
        self.se_reduce_mid = cfg.BACKBONE.MV3.SE_REDUCE_MID
        self.se_divisible = cfg.BACKBONE.MV3.SE_DIVISIBLE
        self.head_use_bias = cfg.BACKBONE.MV3.HEAD_USE_BIAS
        self.force_residual = cfg.BACKBONE.MV3.FORCE_RESIDUAL
        self.sync_se_act = cfg.BACKBONE.MV3.SYNC_SE_ACT
        self.bn_eps = cfg.BACKBONE.BN_EPS
        self.stride = stride
        self.activation = make_act(act=act if layers_cfg[0][0][3] else 'ReLU')

        self.inplanes = make_divisible(layers_cfg[0][0][1] * widen_factor, 8)
        self.conv1 = nn.Conv2d(self.dim_in, self.inplanes, kernel_size=layers_cfg[0][0][0], stride=layers_cfg[0][0][4],
                               padding=layers_cfg[0][0][0] // 2, bias=False)
        self.bn1 = make_norm(self.inplanes, eps=self.bn_eps, norm=norm)

        self.layer0 = self._make_layer(block, layers_cfg[1], dilation=1) if layers_cfg[1][0][0] else None
        self.layer1 = self._make_layer(block, layers_cfg[2], dilation=1)
        self.layer2 = self._make_layer(block, layers_cfg[3], dilation=1)
        self.layer3 = self._make_layer(block, layers_cfg[4], dilation=1)
        self.layer4 = self._make_layer(block, layers_cfg[5], dilation=1)

        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]
        self.spatial_out = self.stage_out_spatial[1:int(math.log(self.stride, 2))]

        del self.last_stage
        del self.avgpool
        del self.conv_out
        del self.fc
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        if self.layer0 is not None:
            x = self.layer0(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# MobileNet V3 Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("mobilenet_v3")
def mobilenet_v3(cfg):
    model = MobileNetV3(cfg)
    if cfg.BACKBONE.MV3.SAME_PAD:
        model = convert_conv2convsamepadding_model(model)
    return model

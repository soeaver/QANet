import math

import torch.nn as nn
import torch.nn.functional as F

import pet.models.imagenet.mobilenet_v3 as mv3
from pet.models.imagenet.utils import make_divisible, convert_conv2convsamepadding_model
from pet.utils.net import make_norm
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class MobileNetV3(mv3.MobileNetV3):
    def __init__(self, norm='bn', activation=mv3.H_Swish, stride=32):
        """ Constructor
        """
        super(MobileNetV3, self).__init__()
        block = mv3.LinearBottleneck
        self.widen_factor = cfg.BACKBONE.MV3.WIDEN_FACTOR
        self.norm = norm
        self.se_reduce_mid = cfg.BACKBONE.MV3.SE_REDUCE_MID
        self.se_divisible = cfg.BACKBONE.MV3.SE_DIVISIBLE
        self.head_use_bias = cfg.BACKBONE.MV3.HEAD_USE_BIAS
        self.force_residual = cfg.BACKBONE.MV3.FORCE_RESIDUAL
        self.sync_se_act = cfg.BACKBONE.MV3.SYNC_SE_ACT
        self.bn_eps = cfg.BACKBONE.BN_EPS
        self.activation_type = activation
        self.stride = stride

        setting = cfg.BACKBONE.MV3.SETTING
        layers_cfg = mv3.MV3_CFG[setting]
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        self.channels = [make_divisible(ch * self.widen_factor, 8) for ch in num_of_channels]
        self.activation = activation() if layers_cfg[0][0][3] else nn.ReLU(inplace=True)

        self.inplanes = make_divisible(layers_cfg[0][0][1] * self.widen_factor, 8)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=layers_cfg[0][0][0], stride=layers_cfg[0][0][4],
                               padding=layers_cfg[0][0][0] // 2, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=self.norm, eps=self.bn_eps)

        self.layer0 = self._make_layer(block, layers_cfg[1], dilation=1) if layers_cfg[1][0][0] else None
        self.layer1 = self._make_layer(block, layers_cfg[2], dilation=1)
        self.layer2 = self._make_layer(block, layers_cfg[3], dilation=1)
        self.layer3 = self._make_layer(block, layers_cfg[4], dilation=1)
        self.layer4 = self._make_layer(block, layers_cfg[5], dilation=1)

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32.]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

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
def mobilenet_v3():
    model = MobileNetV3()
    if cfg.BACKBONE.MV3.SAME_PAD:
        model = convert_conv2convsamepadding_model(model)
    return model

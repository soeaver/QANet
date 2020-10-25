import math
import torch.nn as nn

import lib.backbone.mobilenet_v2 as mv2
from lib.layers import InvertedResidual, make_act, make_norm
from lib.utils.net import make_divisible

from instance.modeling import registry


class MobileNetV2(mv2.MobileNetV2):
    def __init__(self, cfg, act="ReLU6", stride=32):
        """ Constructor
        """
        super(MobileNetV2, self).__init__()
        self.dim_in = 3
        self.spatial_in = [1]

        block = InvertedResidual
        widen_factor = cfg.BACKBONE.MV2.WIDEN_FACTOR
        layers_cfg = mv2.MV2_CFG['A']
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        channels = [make_divisible(ch * widen_factor, 8) for ch in num_of_channels]
        norm = cfg.BACKBONE.MV2.NORM

        self.widen_factor = widen_factor
        self.channels = channels
        self.norm = norm
        self.act = act
        self.stride = stride
        self.activation = make_act(act=act)

        self.inplanes = make_divisible(layers_cfg[0][0][1] * widen_factor, 8)
        self.conv1 = nn.Conv2d(self.dim_in, self.inplanes, kernel_size=layers_cfg[0][0][0], stride=layers_cfg[0][0][4],
                               padding=layers_cfg[0][0][0] // 2, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=norm)

        self.layer0 = self._make_layer(block, layers_cfg[1], dilation=1)
        self.layer1 = self._make_layer(block, layers_cfg[2], dilation=1)
        self.layer2 = self._make_layer(block, layers_cfg[3], dilation=1)
        self.layer3 = self._make_layer(block, layers_cfg[4], dilation=1)
        self.layer4 = self._make_layer(block, layers_cfg[5], dilation=1)

        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]
        self.spatial_out = self.stage_out_spatial[1:int(math.log(self.stride, 2))]

        del self.conv_out
        del self.bn_out
        del self.avgpool
        del self.fc
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer0(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# MobileNet V2 Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("mobilenet_v2")
def mobilenet_v2(cfg):
    model = MobileNetV2(cfg)
    return model

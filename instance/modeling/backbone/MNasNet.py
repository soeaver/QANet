import math

import torch.nn as nn

import pet.models.imagenet.mnasnet as mnas
from pet.models.imagenet.utils import make_divisible
from pet.utils.net import make_norm
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class MNasNet(mnas.MNasNet):
    def __init__(self, norm='bn', activation=nn.ReLU, stride=32):
        """ Constructor
        """
        super(MnasNet, self).__init__()
        block = mnas.LinearBottleneck
        self.widen_factor = cfg.BACKBONE.MNAS.WIDEN_FACTOR
        self.norm = norm
        self.activation_type = activation
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()
        self.stride = stride

        setting = cfg.BACKBONE.MNAS.SETTING
        layers_cfg = mnas.MNAS_CFG[setting]
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        self.channels = [make_divisible(ch * self.widen_factor, 8) for ch in num_of_channels]

        self.inplanes = make_divisible(layers_cfg[0][0][1] * self.widen_factor, 8)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=layers_cfg[0][0][0], stride=layers_cfg[0][0][4],
                               padding=layers_cfg[0][0][0] // 2, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=self.norm)

        self.layer0 = self._make_layer(block, layers_cfg[1], dilation=1)
        self.layer1 = self._make_layer(block, layers_cfg[2], dilation=1)
        self.layer2 = self._make_layer(block, layers_cfg[3], dilation=1)
        self.layer3 = self._make_layer(block, layers_cfg[4], dilation=1)
        self.layer4 = self._make_layer(block, layers_cfg[5], dilation=1)

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32.]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

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
# MNasNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("mnasnet")
def mnasnet():
    model = MNasNet()
    return model

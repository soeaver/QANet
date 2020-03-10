import math

import torch.nn as nn

import pet.models.imagenet.fbnet as fb
from pet.models.imagenet.utils import make_divisible
from pet.utils.net import make_norm
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class FBNet(fb.FBNet):
    def __init__(self, norm='bn', activation=nn.ReLU, stride=32):
        """ Constructor
        """
        super(FBNet, self).__init__()
        block = fb.LinearBottleneck
        self.widen_factor = cfg.BACKBONE.FB.WIDEN_FACTOR
        self.norm = norm
        self.bn_eps = cfg.BACKBONE.BN_EPS
        self.activation_type = activation
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()
        self.stride = stride

        setting = cfg.BACKBONE.FB.SETTING
        layers_cfg = fb.FBNET_CFG[setting]
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        self.channels = [make_divisible(ch * self.widen_factor, 8) for ch in num_of_channels]

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
            
        del self.conv_out
        del self.bn_out
        del self.avgpool
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
# FBNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("fbnet")
def fbnet():
    model = FBNet()
    return model

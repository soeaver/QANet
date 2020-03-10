import math

import torch.nn as nn

import pet.models.imagenet.mobilenet_v1 as mv1
import pet.models.ops as ops
from pet.models.imagenet.utils import make_divisible
from pet.utils.net import make_norm
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class MobileNetV1(mv1.MobileNetV1):
    def __init__(self, norm='bn', activation=nn.ReLU, stride=32):
        """ Constructor
        """
        super(MobileNetV1, self).__init__()
        block = mv1.BasicBlock
        self.use_se = cfg.BACKBONE.MV1.USE_SE
        self.norm = norm
        self.activation_type = activation
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()
        self.stride = stride

        layers = cfg.BACKBONE.MV1.LAYERS
        kernel = cfg.BACKBONE.MV1.KERNEL
        num_of_channels = cfg.BACKBONE.MV1.NUM_CHANNELS
        channels = [make_divisible(ch * cfg.BACKBONE.MV1.WIDEN_FACTOR, 8) for ch in num_of_channels]
        self.channels = channels

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = make_norm(channels[0], norm=self.norm)
        self.conv2 = nn.Conv2d(channels[0], channels[0], kernel_size=kernel, stride=1, padding=kernel // 2,
                               groups=channels[0], bias=False)
        self.bn2 = make_norm(channels[0], norm=self.norm)
        self.conv3 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = make_norm(channels[1], norm=self.norm)
        self.inplanes = channels[1]

        self.layer1 = self._make_layer(block, channels[2], layers[0], stride=2, dilation=1, kernel=kernel)
        self.layer2 = self._make_layer(block, channels[3], layers[1], stride=2, dilation=1, kernel=kernel)
        self.layer3 = self._make_layer(block, channels[4], layers[2], stride=2, dilation=1, kernel=kernel)
        self.layer4 = self._make_layer(block, channels[5], layers[3], stride=2, dilation=1, kernel=kernel)

        self.dropblock = ops.DropBlock2D(keep_prob=0.9, block_size=7) if cfg.BACKBONE.MV1.USE_DP else None

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32.]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        del self.avgpool
        del self.fc
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        if self.dropblock is not None:
            x4 = self.dropblock(x4)
        x5 = self.layer4(x4)
        if self.dropblock is not None:
            x5 = self.dropblock(x5)
            
        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# MobileNet V1 Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("mobilenet_v1")
def mobilenet_v1():
    model = MobileNetV1()
    return model

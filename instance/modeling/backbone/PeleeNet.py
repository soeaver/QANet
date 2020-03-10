import math

import torch.nn as nn

import pet.models.imagenet.peleenet as pelee
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class PeleeNet(pelee.PeleeNet):
    def __init__(self, norm='bn', activation=nn.ReLU, stride=32):
        """ Constructor
        """
        super(PeleeNet, self).__init__()
        block = pelee.DenseBasicBlock
        self.growth_rate = cfg.BACKBONE.PELEE.GROWTH_RATE
        self.inplanes = cfg.BACKBONE.PELEE.NUM_INIT
        self.norm = norm
        self.activation_type = activation
        self.stride = stride

        layers = cfg.BACKBONE.PELEE.LAYERS
        bottleneck_width = cfg.BACKBONE.PELEE.BOTTLENECK_WIDTH

        # check growth rate and bottleneck width
        if type(self.growth_rate) is list:
            growth_rates = self.growth_rate
            assert len(growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [self.growth_rate] * 4

        if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
            assert len(bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [bottleneck_width] * 4
        self.channels = []

        self.layer0 = pelee.StemBlock(3, self.inplanes, conv1_stride=2, norm=self.norm, activation=self.activation_type)
        self.channels.append(self.inplanes)

        # Dense-Block 1 and transition
        self.layer1 = self._make_layer(block, layers[0], bottleneck_widths[0])
        self.inplanes = self.inplanes + layers[0] * growth_rates[0]
        self.channels.append(self.inplanes)
        self.translayer1 = self._make_transition()

        # Dense-Block 2 and transition
        self.layer2 = self._make_layer(block, layers[1], bottleneck_widths[1])
        self.inplanes = self.inplanes + layers[1] * growth_rates[1]
        self.channels.append(self.inplanes)
        self.translayer2 = self._make_transition()

        # Dense-Block 3 and transition
        self.layer3 = self._make_layer(block, layers[2], bottleneck_widths[2])
        self.inplanes = self.inplanes + layers[2] * growth_rates[2]
        self.channels.append(self.inplanes)
        self.translayer3 = self._make_transition()

        # Dense-Block 4
        self.layer4 = self._make_layer(block, layers[3], bottleneck_widths[3])
        self.inplanes = self.inplanes + layers[3] * growth_rates[3]
        self.channels.append(self.inplanes)

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32.]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        del self.translayer4
        del self.avgpool
        del self.fc
        self._init_weights()

    def forward(self, x):
        x = self.layer0(x)
        
        x2 = self.layer1(x)
        x = self.translayer1(x2)
        x3 = self.layer2(x)
        x = self.translayer2(x3)
        x4 = self.layer3(x)
        x = self.translayer3(x4)
        x5 = self.layer4(x)

        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# PeleeNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("peleenet")
def peleenet():
    model = PeleeNet()
    return model

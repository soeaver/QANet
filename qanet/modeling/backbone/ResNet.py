import math
import torch.nn as nn

import lib.backbone.resnet as res
from lib.layers import AlignedBottleneck, BasicBlock, Bottleneck, make_norm
from lib.utils.net import convert_conv2convws_model

from qanet.modeling import registry


class ResNet(res.ResNet):
    def __init__(self, cfg, stride=32):
        """ Constructor
        """
        super(ResNet, self).__init__()
        self.dim_in = 3
        self.spatial_in = [1]

        if cfg.BACKBONE.RESNET.USE_ALIGN:
            block = AlignedBottleneck
        else:
            if cfg.BACKBONE.RESNET.BOTTLENECK:
                block = Bottleneck  # not use the original Bottleneck module
            else:
                block = BasicBlock

        stem_width = cfg.BACKBONE.RESNET.STEM_WIDTH
        layers = cfg.BACKBONE.RESNET.LAYERS[:int(math.log(stride, 2)) - 1]
        stage_with_conv = cfg.BACKBONE.RESNET.STAGE_WITH_CONV
        norm = cfg.BACKBONE.RESNET.NORM
        stage_with_ctx = cfg.BACKBONE.RESNET.STAGE_WITH_CTX

        self.expansion = block.expansion
        self.use_3x3x3stem = cfg.BACKBONE.RESNET.USE_3x3x3HEAD
        self.stride_3x3 = cfg.BACKBONE.RESNET.STRIDE_3X3
        self.avg_down = cfg.BACKBONE.RESNET.AVG_DOWN
        self.base_width = cfg.BACKBONE.RESNET.WIDTH
        self.radix = cfg.BACKBONE.RESNET.RADIX
        self.norm = norm
        self.stride = stride

        self.inplanes = stem_width
        if not self.use_3x3x3stem:
            self.conv1 = nn.Conv2d(self.dim_in, self.inplanes, 7, 2, 3, bias=False)
            self.bn1 = make_norm(self.inplanes, norm=norm.replace('Mix', ''))
        else:
            self.conv1 = nn.Conv2d(self.dim_in, self.inplanes // 2, 3, 2, 1, bias=False)
            self.bn1 = make_norm(self.inplanes // 2, norm=norm.replace('Mix', ''))
            self.conv2 = nn.Conv2d(self.inplanes // 2, self.inplanes // 2, 3, 1, 1, bias=False)
            self.bn2 = make_norm(self.inplanes // 2, norm=norm.replace('Mix', ''))
            self.conv3 = nn.Conv2d(self.inplanes // 2, self.inplanes, 3, 1, 1, bias=False)
            self.bn3 = make_norm(self.inplanes, norm=norm.replace('Mix', ''))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1, conv=stage_with_conv[0], ctx=stage_with_ctx[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2, conv=stage_with_conv[1], ctx=stage_with_ctx[1])
        self.layer3 = self._make_layer(block, 256, layers[2], 2, conv=stage_with_conv[2], ctx=stage_with_ctx[2])
        self.layer4 = self._make_layer(block, 512, layers[3], 2, conv=stage_with_conv[3], ctx=stage_with_ctx[3])

        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]
        self.spatial_out = self.stage_out_spatial[1:int(math.log(self.stride, 2))]

        del self.avgpool
        del self.fc
        self._init_weights()

    def forward(self, x):
        if not self.use_3x3x3stem:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# ResNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("resnet")
def resnet(cfg):
    stride = cfg.BACKBONE.RESNET.STRIDE
    model = ResNet(cfg, stride=stride)
    return model

import math

import torch.nn as nn

import pet.models.imagenet.resnet as res
from pet.utils.net import make_norm
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


def get_norm():
    norm = 'bn'
    if cfg.BACKBONE.RESNET.USE_GN:
        norm = 'gn'
    return norm


class ResNet(res.ResNet):
    def __init__(self, norm='bn', stride=32):
        """ Constructor
        """
        super(ResNet, self).__init__()
        if cfg.BACKBONE.RESNET.USE_ALIGN:
            block = res.AlignedBottleneck
        else:
            if cfg.BACKBONE.RESNET.BOTTLENECK:
                block = res.Bottleneck  # not use the original Bottleneck module
            else:
                block = res.BasicBlock
        self.expansion = block.expansion
        self.stride_3x3 = cfg.BACKBONE.RESNET.STRIDE_3X3
        self.avg_down = cfg.BACKBONE.RESNET.AVG_DOWN
        self.norm = norm
        self.stride = stride

        if self.stride == 8:
            strides = (1, 1)
            dilations = (2, 4)
        elif self.stride == 16:
            strides = (2, 1)
            dilations = (1, 2)
        else:
            strides = (2, 2)
            dilations = (1, 1)

        layers = cfg.BACKBONE.RESNET.LAYERS
        self.base_width = cfg.BACKBONE.RESNET.WIDTH
        stage_with_context = cfg.BACKBONE.RESNET.STAGE_WITH_CONTEXT
        self.ctx_ratio = cfg.BACKBONE.RESNET.CTX_RATIO
        stage_with_conv = cfg.BACKBONE.RESNET.STAGE_WITH_CONV

        self.inplanes = 64
        self.use_3x3x3stem = cfg.BACKBONE.RESNET.USE_3x3x3HEAD
        if not self.use_3x3x3stem:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
            self.bn1 = make_norm(self.inplanes, norm=self.norm)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes // 2, 3, 2, 1, bias=False)
            self.bn1 = make_norm(self.inplanes // 2, norm=self.norm)
            self.conv2 = nn.Conv2d(self.inplanes // 2, self.inplanes // 2, 3, 1, 1, bias=False)
            self.bn2 = make_norm(self.inplanes // 2, norm=self.norm)
            self.conv3 = nn.Conv2d(self.inplanes // 2, self.inplanes, 3, 1, 1, bias=False)
            self.bn3 = make_norm(self.inplanes, norm=self.norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1, conv=stage_with_conv[0], context=stage_with_context[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2, conv=stage_with_conv[1], context=stage_with_context[1])
        self.layer3 = self._make_layer(block, 256, layers[2], strides[0], dilations[0], conv=stage_with_conv[2],
                                       context=stage_with_context[2])
        self.layer4 = self._make_layer(block, 512, layers[3], strides[1], dilations[1], conv=stage_with_conv[3],
                                       context=stage_with_context[3])

        # self.dim_out = self.stage_out_dim[int(math.log(self.stride, 2)) - 1:]
        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16. * dilations[0], 1 / 32. * dilations[1]]
        self.dim_out = self.stage_out_dim[1:]

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
def resnet():
    stride = cfg.BACKBONE.RESNET.STRIDE
    model = ResNet(norm=get_norm(), stride=stride)
    return model

import math

import torch.nn as nn

import pet.models.imagenet.crossnet as cx
from pet.utils.net import make_norm
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class CrossNet(cx.CrossNet):
    def __init__(self, norm='bn', stride=32):
        """ Constructor
        """
        super(CrossNet, self).__init__()
        block = cx.CrossPod
        self.use_se = use_se
        self.norm = norm
        self.stride = stride

        layers = cfg.BACKBONE.CX.LAYERS
        base_width = cfg.BACKBONE.CX.WIDTH
        expansion = cfg.BACKBONE.CX.EXPANSION
        kernel = cfg.BACKBONE.CX.KERNEL
        groups = cfg.BACKBONE.CX.GROUPS
        depth = cfg.BACKBONE.CX.DEPTH

        head_dim = 24
        if base_width == 80:
            head_dim = 32
        elif base_width == 90 or base_width == 100:
            head_dim = 48
        self.channels = [head_dim, int(base_width * 1 * expansion), int(base_width * 2 * expansion),
                         int(base_width * 4 * expansion), int(base_width * 8 * expansion)]
        
        self.inplanes = head_dim
        self.conv1 = nn.Conv2d(3, head_dim, 3, 2, 1, bias=False)
        self.bn1 = make_norm(head_dim, norm=self.norm)
        self.conv2 = nn.Conv2d(head_dim, head_dim, kernel_size=1, stride=1, bias=False)
        self.bn2 = make_norm(head_dim, norm=self.norm)
        self.conv3 = nn.Conv2d(head_dim, head_dim, kernel_size=kernel, stride=2, padding=kernel // 2,
                               groups=head_dim, bias=False)
        self.bn3 = make_norm(head_dim, norm=self.norm)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, base_width, layers[0], expansion, 1, 1, kernel, groups, depth)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], expansion, 2, 1, kernel, groups, depth)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], expansion, 2, 1, kernel, groups, depth)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], expansion, 2, 1, kernel, groups, depth)

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
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# CrossNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("crossnet")
def crossnet():
    model = CrossNet()
    return model

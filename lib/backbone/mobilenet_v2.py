"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen, et.al. (2018 CVPR). 
Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation. 
Copyright (c) Yang Lu, 2018
"""
import torch.nn as nn
import torch.nn.functional as F

from lib.layers import InvertedResidual, make_act, make_norm
from lib.utils.net import make_divisible

MV2_CFG = {
    # 0,      1,           2,        3,   4,      5,      6,      7
    # kernel, out_channel, se_ratio, act, stride, group1, group2, t
    'A': [
        [[3, 32, 0, 0, 2, 0, 0, 0]],  # stem (conv1)
        [[3, 16, 0, 0, 1, 1, 1, 1]],  # layer0
        [[3, 24, 0, 0, 2, 1, 1, 6],  # layer1
         [3, 24, 0, 0, 1, 1, 1, 6]],
        [[3, 32, 0, 0, 2, 1, 1, 6],  # layer2
         [3, 32, 0, 0, 1, 1, 1, 6],
         [3, 32, 0, 0, 1, 1, 1, 6]],
        [[3, 64, 0, 0, 2, 1, 1, 6],  # layer3
         [3, 64, 0, 0, 1, 1, 1, 6],
         [3, 64, 0, 0, 1, 1, 1, 6],
         [3, 64, 0, 0, 1, 1, 1, 6],
         [3, 96, 0, 0, 1, 1, 1, 6],
         [3, 96, 0, 0, 1, 1, 1, 6],
         [3, 96, 0, 0, 1, 1, 1, 6]],
        [[3, 160, 0, 0, 2, 1, 1, 6],  # layer4
         [3, 160, 0, 0, 1, 1, 1, 6],
         [3, 160, 0, 0, 1, 1, 1, 6],
         [3, 320, 0, 0, 1, 1, 1, 6]],
        [[1, 1280, 0, 0, 1, 0, 0, 0]]  # head (conv_out)
    ]
}


class MobileNetV2(nn.Module):
    def __init__(self, widen_factor=1.0, norm='BN', act='ReLU6', drop_rate=0.0, num_classes=1000):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        self.widen_factor = widen_factor
        self.norm = norm
        self.act = act
        self.drop_rate = drop_rate

        layers_cfg = MV2_CFG['A']
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        self.channels = [make_divisible(ch * self.widen_factor, 8) for ch in num_of_channels]
        self.activation = make_act(act=act)

        self.inplanes = make_divisible(layers_cfg[0][0][1] * self.widen_factor, 8)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=layers_cfg[0][0][0], stride=layers_cfg[0][0][4],
                               padding=layers_cfg[0][0][0] // 2, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=norm)

        self.layer0 = self._make_layer(block, layers_cfg[1], dilation=1)
        self.layer1 = self._make_layer(block, layers_cfg[2], dilation=1)
        self.layer2 = self._make_layer(block, layers_cfg[3], dilation=1)
        self.layer3 = self._make_layer(block, layers_cfg[4], dilation=1)
        self.layer4 = self._make_layer(block, layers_cfg[5], dilation=1)

        out_ch = layers_cfg[-1][-1][1]
        self.conv_out = nn.Conv2d(self.inplanes, out_ch, kernel_size=layers_cfg[-1][-1][0],
                                  stride=layers_cfg[-1][-1][4], padding=layers_cfg[-1][-1][0] // 2, bias=False)
        self.bn_out = make_norm(out_ch, norm=norm)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_ch, num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return self.channels

    @property
    def stage_out_spatial(self):
        return [1 / 2., 1 / 4., 1 / 8., 1 / 16., 1 / 32.]

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, lc, dilation=1):
        layers = []
        for i in range(0, len(lc)):
            layers.append(
                block(self.inplanes, make_divisible(lc[i][1] * self.widen_factor, 8), kernel=lc[i][0],
                      stride=lc[i][4], dilation=dilation, groups=(lc[i][5], lc[i][6]),
                      t=lc[i][7], norm=self.norm, act=self.act, se_ratio=lc[i][2])
            )
            self.inplanes = make_divisible(lc[i][1] * self.widen_factor, 8)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.activation(x)

        x = self.avgpool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

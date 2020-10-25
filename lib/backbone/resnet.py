"""
Creates a ResNet Model as defined in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2015 CVPR). 
Deep Residual Learning for Image Recognition. 
Copyright (c) Yang Lu, 2017
"""
import torch.nn as nn

import lib.ops as ops
from lib.layers import AlignedBottleneck, BasicBlock, Bottleneck, make_norm


class ResNet(nn.Module):
    def __init__(self, bottleneck=True, aligned=False, use_3x3x3stem=False, stride_3x3=False, avg_down=False,
                 stem_width=64, base_width=64, layers=(3, 4, 6, 3), radix=1,
                 stage_with_conv=('Conv2d', 'Conv2d', 'Conv2d', 'Conv2d'), norm='BN', stage_with_ctx=('', '', '', ''),
                 num_classes=1000):
        """ Constructor
        Args:
            layers: config of layers, e.g., (3, 4, 23, 3)
            num_classes: number of classes
        """
        super(ResNet, self).__init__()
        if aligned:
            block = AlignedBottleneck
        else:
            if bottleneck:
                block = Bottleneck
            else:
                block = BasicBlock
        self.expansion = block.expansion
        self.use_3x3x3stem = use_3x3x3stem
        self.stride_3x3 = stride_3x3
        self.avg_down = avg_down
        self.base_width = base_width
        self.radix = radix
        self.norm = norm

        self.inplanes = stem_width
        if not self.use_3x3x3stem:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
            self.bn1 = make_norm(self.inplanes, norm=norm.replace('Mix', ''))
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes // 2, 3, 2, 1, bias=False)
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

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return [64, 64 * self.expansion, 128 * self.expansion, 256 * self.expansion, 512 * self.expansion]

    @property
    def stage_out_spatial(self):
        return [1 / 2., 1 / 4., 1 / 8., 1 / 16., 1 / 32.]

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if not isinstance(m, (ops.MixtureBatchNorm2d, ops.MixtureGroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.0001)
                nn.init.constant_(m.bias, 0)
        # zero init deform conv offset
        for m in self.modules():
            if isinstance(m, ops.DeformConvPack):
                nn.init.constant_(m.conv_offset.weight, 0)
                nn.init.constant_(m.conv_offset.bias, 0)
            if isinstance(m, ops.ModulatedDeformConvPack):
                nn.init.constant_(m.conv_offset_mask.weight, 0)
                nn.init.constant_(m.conv_offset_mask.bias, 0)
        # zero gamma for last bn of each block
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, AlignedBottleneck):
                nn.init.constant_(m.bn.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, conv='Conv2d', ctx=''):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNet
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    make_norm(planes * block.expansion, norm=self.norm.replace('Mix', '')),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    make_norm(planes * block.expansion, norm=self.norm.replace('Mix', '')),
                )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.base_width, 1, stride, dilation, radix=self.radix, downsample=downsample,
                  stride_3x3=self.stride_3x3, conv=conv, norm=self.norm, ctx=ctx)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, self.base_width, 1, 1, dilation, radix=self.radix, downsample=None,
                      stride_3x3=self.stride_3x3, conv=conv, norm=self.norm, ctx=ctx)
            )

        return nn.Sequential(*layers)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

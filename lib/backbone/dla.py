"""
Creates a DLA Model as defined in:
Fisher Yu, Dequan Wang,  Evan Shelhamer, Trevor Darrell. (2018 CVPR).
Deep Layer Aggregation.
Copyright (c) Yang Lu, 2019
"""
import torch
import torch.nn as nn

import lib.ops as ops
from lib.layers import make_conv, make_norm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, conv='Conv2d', norm='BN'):
        super(BasicBlock, self).__init__()
        self.conv1 = make_conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False, conv=conv)
        self.bn1 = make_norm(planes, norm=norm)
        self.conv2 = make_conv(planes, planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation,
                               bias=False, conv=conv)
        self.bn2 = make_norm(planes, norm=norm)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1, conv='Conv2d', norm='BN'):
        super(Bottleneck, self).__init__()
        bottle_planes = planes // self.expansion

        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = make_norm(bottle_planes, norm=norm)
        self.conv2 = make_conv(bottle_planes, bottle_planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False, conv=conv)
        self.bn2 = make_norm(bottle_planes, norm=norm)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = make_norm(planes, norm=norm)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1, conv='Conv2d', norm='BN'):
        super(BottleneckX, self).__init__()
        bottle_planes = planes * self.cardinality // 32

        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = make_norm(bottle_planes, norm=norm)
        self.conv2 = make_conv(bottle_planes, bottle_planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, groups=self.cardinality, bias=False, conv=conv)
        self.bn2 = make_norm(bottle_planes, norm=norm)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = make_norm(planes, norm=norm)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual, norm='BN'):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = make_norm(out_channels, norm=norm)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0,
                 root_kernel_size=1, dilation=1, root_residual=False, conv='Conv2d', norm='BN'):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation, conv=conv, norm=norm)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation, conv=conv, norm=norm)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual,
                              conv=conv, norm=norm)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual,
                              conv=conv, norm=norm)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual, norm=norm)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels and levels == 1:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                make_norm(out_channels, norm=norm)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels=(1, 1, 1, 2, 2, 1), channels=(16, 32, 64, 128, 256, 512), cardinality=32,
                 bottleneck=False, bottle_x=False, residual_root=False, return_levels=False,
                 stage_with_conv=('Conv2d', 'Conv2d', 'Conv2d', 'Conv2d'), norm='BN', num_classes=1000):
        """ Constructor
        Args:
            levels: config of levels, e.g., (1, 1, 1, 2, 2, 1)
            channels: config of channels, e.g., (16, 32, 64, 128, 256, 512)
            num_classes: number of classes
        """
        super(DLA, self).__init__()
        if bottleneck:
            if bottle_x:
                block = BottleneckX
                block.cardinality = cardinality
            else:
                block = Bottleneck
        else:
            block = BasicBlock
        self.channels = channels
        self.return_levels = return_levels
        self.norm = norm

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = make_norm(channels[0], norm=norm)
        self.relu = nn.ReLU(inplace=True)

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], stride=2, level_root=False,
                           root_residual=residual_root, conv=stage_with_conv[0], norm=norm)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], stride=2, level_root=True,
                           root_residual=residual_root, conv=stage_with_conv[1], norm=norm)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], stride=2, level_root=True,
                           root_residual=residual_root, conv=stage_with_conv[2], norm=norm)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], stride=2, level_root=True,
                           root_residual=residual_root, conv=stage_with_conv[3], norm=norm)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return self.channels[1:]

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

    def _make_conv_level(self, inplanes, planes, levels, stride=1, dilation=1):
        modules = []
        for i in range(levels):
            modules.extend(
                [nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, dilation=dilation,
                           padding=dilation, bias=False),
                 make_norm(planes, norm=self.norm),
                 nn.ReLU(inplace=True)]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

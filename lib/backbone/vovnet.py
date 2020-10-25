"""
Creates a ResNet Model as defined in:
Youngwan Lee, Joong-won Hwang, Sangrok Lee, Yuseok Bae. (2019 CVPRW).
An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection.
Copyright (c) Yang Lu, 2019
"""
import torch
import torch.nn as nn

import lib.ops as ops
from lib.layers import make_conv, make_norm


class eSE(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = ops.H_Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class OSABlock(nn.Module):
    def __init__(self, inplanes, planes, outplanes, num_conv=5, dilation=1, conv='Conv2d', norm='BN', use_dw=False,
                 use_eSE=False, identity=False):
        super(OSABlock, self).__init__()
        self.use_dw = use_dw
        self.use_eSE = use_eSE
        self.identity = identity
        self.is_reduced = False
        self.layers = nn.ModuleList()
        dim_in = inplanes
        if self.use_dw and dim_in != planes:
            self.is_reduced = True
            self.conv_reduction = nn.Sequential(
                nn.Conv2d(dim_in, planes, kernel_size=1, bias=False),
                make_norm(planes, norm=norm),
                nn.ReLU(inplace=True)
            )
            dim_in = planes
        for i in range(num_conv):
            if self.use_dw:
                self.layers.append(
                    nn.Sequential(
                        make_conv(dim_in, planes, kernel_size=3, stride=1, groups=planes, dilation=dilation,
                                  padding=dilation, bias=False, conv=conv),
                        nn.Conv2d(planes, planes, kernel_size=1, bias=False),
                        make_norm(planes, norm=norm),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        make_conv(dim_in, planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation,
                                  bias=False, conv=conv),
                        make_norm(planes, norm=norm),
                        nn.ReLU(inplace=True)
                    )
                )
            dim_in = planes

        # feature aggregation
        dim_in = inplanes + num_conv * planes
        self.concat = nn.Sequential(
            nn.Conv2d(dim_in, outplanes, kernel_size=1, stride=1, bias=False),
            make_norm(outplanes, norm=norm),
            nn.ReLU(inplace=True)
        )

        if self.use_eSE:
            self.ese = eSE(outplanes)

    def forward(self, x):
        identity_feat = x
        output = [x]
        if self.use_dw and self.is_reduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.use_eSE:
            xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class VoVNet(nn.Module):
    def __init__(self, use_dw=False, use_eSE=False, base_width=64, stage_dims=(128, 160, 192, 224),
                 concat_dims=(256, 512, 768, 1024), layers=(1, 1, 2, 2), num_conv=5,
                 stage_with_conv=('Conv2d', 'Conv2d', 'Conv2d', 'Conv2d'), norm='BN', num_classes=1000):
        """ Constructor
        Args:
            layers: config of layers, e.g., (1, 1, 2, 2)
            num_classes: number of classes
        """
        super(VoVNet, self).__init__()
        block = OSABlock
        self.use_dw = use_dw
        self.use_eSE = use_eSE
        self.num_conv = num_conv
        self.norm = norm
        self.channels = [base_width] + list(concat_dims)

        self.inplanes = base_width
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 2, 1, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=norm)
        if self.use_dw:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes, 3, 1, 1, groups=self.inplanes, bias=False),
                nn.Conv2d(self.inplanes, self.inplanes, 1, 1, 0, bias=False)
            )
            self.bn2 = make_norm(self.inplanes, norm=norm)
            self.conv3 = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes, 3, 2, 1, groups=self.inplanes, bias=False),
                nn.Conv2d(self.inplanes, self.inplanes, 1, 1, 0, bias=False)
            )
            self.bn3 = make_norm(self.inplanes, norm=norm)
        else:
            self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, 3, 1, 1, bias=False)
            self.bn2 = make_norm(self.inplanes, norm=norm)
            self.conv3 = nn.Conv2d(self.inplanes, self.inplanes * 2, 3, 2, 1, bias=False)
            self.bn3 = make_norm(self.inplanes * 2, norm=norm)
            self.inplanes = self.inplanes * 2
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, stage_dims[0], concat_dims[0], layers[0], 1, conv=stage_with_conv[0])
        self.layer2 = self._make_layer(block, stage_dims[1], concat_dims[1], layers[1], 2, conv=stage_with_conv[1])
        self.layer3 = self._make_layer(block, stage_dims[2], concat_dims[2], layers[2], 2, conv=stage_with_conv[2])
        self.layer4 = self._make_layer(block, stage_dims[3], concat_dims[3], layers[3], 2, conv=stage_with_conv[3])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inplanes, num_classes)

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
        # zero init deform conv offset
        for m in self.modules():
            if isinstance(m, ops.DeformConvPack):
                nn.init.constant_(m.conv_offset.weight, 0)
                nn.init.constant_(m.conv_offset.bias, 0)
            if isinstance(m, ops.ModulatedDeformConvPack):
                nn.init.constant_(m.conv_offset_mask.weight, 0)
                nn.init.constant_(m.conv_offset_mask.bias, 0)

    def _make_layer(self, block, planes, outplanes, blocks, stride=1, dilation=1, conv='Conv2d'):
        layers = []
        if stride != 1:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        layers.append(
            block(self.inplanes, planes, outplanes, self.num_conv, dilation, conv, self.norm, self.use_dw, self.use_eSE)
        )
        self.inplanes = outplanes
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, outplanes, self.num_conv, dilation, conv, self.norm, self.use_dw,
                      self.use_eSE, True)
            )

        return nn.Sequential(*layers)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

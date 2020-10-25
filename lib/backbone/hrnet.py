"""
Creates a HRNet Model as defined in:
Ke Sun, Bin Xiao, Dong Liu and Jingdong Wang. (2019 CVPR).
Deep High-Resolution Representation Learning for Human Pose Estimation.
Copyright (c) Yang Lu, 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.ops as ops
from lib.layers import BasicBlock, Bottleneck, make_norm


class StageModule(nn.Module):
    def __init__(self, block, planes, stage=2, output_branches=2, conv='Conv2d', norm='BN', ctx='', use_global=False):
        super(StageModule, self).__init__()
        self.use_global = use_global

        self.branches = nn.ModuleList()
        for i in range(stage):
            w = planes * (2 ** i)
            branch = nn.Sequential(
                block(w, w, stride_3x3=True, conv=conv, norm=norm, ctx=ctx),
                block(w, w, stride_3x3=True, conv=conv, norm=norm, ctx=ctx),
                block(w, w, stride_3x3=True, conv=conv, norm=norm, ctx=ctx),
                block(w, w, stride_3x3=True, conv=conv, norm=norm, ctx=ctx),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        if self.use_global:
            self.global_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(planes * (2 ** j), planes * (2 ** i), 1, 1, 0, bias=False),
                        make_norm(planes * (2 ** i), norm=norm.replace('Mix', '')),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(planes * (2 ** j), planes * (2 ** j), 3, 2, 1, bias=False),
                            make_norm(planes * (2 ** j), norm=norm.replace('Mix', '')),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(planes * (2 ** j), planes * (2 ** i), 3, 2, 1, bias=False),
                        make_norm(planes * (2 ** i), norm=norm.replace('Mix', '')),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
            if self.use_global:
                sum_planes = sum([planes * (2 ** k) for k in range(stage)])
                self.global_layers.append(
                    nn.Sequential(
                        nn.Conv2d(sum_planes, planes * (2 ** i), 1, 1, 0, bias=False),
                        make_norm(planes * (2 ** i), norm=norm.replace('Mix', '')),
                        nn.Sigmoid()
                    )
                )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        if self.use_global:
            x_global = [F.adaptive_avg_pool2d(b, 1) for b in x]
            x_global = torch.cat(tuple(x_global), 1)

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])
            if self.use_global:
                x_fused[i] = x_fused[i] * self.global_layers[i](x_global)

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class HRNet(nn.Module):
    def __init__(self, avg_down=False, use_global=False, base_width=32, radix=1,
                 stage_with_conv=('Conv2d', 'Conv2d', 'Conv2d', 'Conv2d'), norm='BN', stage_with_ctx=('', '', '', ''),
                 num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(HRNet, self).__init__()
        block_1 = Bottleneck
        block_2 = BasicBlock
        self.avg_down = avg_down
        self.base_width = base_width
        self.radix = radix
        self.norm = norm
        self.head_dim = (32, 64, 128, 256)

        self.inplanes = 64  # default 64
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = make_norm(64, norm=norm.replace('Mix', ''))
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = make_norm(64, norm=norm.replace('Mix', ''))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_1, 64, 4, 1, conv=stage_with_conv[0], ctx=stage_with_ctx[0])
        self.transition1 = self._make_transition(index=1, stride=2)  # Fusion layer 1: create full and 1/2 resolution

        self.stage2 = nn.Sequential(
            StageModule(block_2, base_width, 2, 2, stage_with_conv[1], norm, stage_with_ctx[1], False),
        )  # Stage 2 with 1 group of block modules, which has 2 branches
        self.transition2 = self._make_transition(index=2, stride=2)  # Fusion layer 2: create 1/4 resolution

        self.stage3 = nn.Sequential(
            StageModule(block_2, base_width, 3, 3, stage_with_conv[2], norm, stage_with_ctx[2], use_global),
            StageModule(block_2, base_width, 3, 3, stage_with_conv[2], norm, stage_with_ctx[2], use_global),
            StageModule(block_2, base_width, 3, 3, stage_with_conv[2], norm, stage_with_ctx[2], use_global),
            StageModule(block_2, base_width, 3, 3, stage_with_conv[2], norm, stage_with_ctx[2], use_global),
        )  # Stage 3 with 4 groups of block modules, which has 3 branches
        self.transition3 = self._make_transition(index=3, stride=2)  # Fusion layer 3: create 1/8 resolution

        self.stage4 = nn.Sequential(
            StageModule(block_2, base_width, 4, 4, stage_with_conv[3], norm, stage_with_ctx[3], use_global),
            StageModule(block_2, base_width, 4, 4, stage_with_conv[3], norm, stage_with_ctx[3], use_global),
            StageModule(block_2, base_width, 4, 4, stage_with_conv[3], norm, stage_with_ctx[3], use_global),
        )  # Stage 4 with 3 groups of block modules, which has 4 branches

        pre_stage_channels = [base_width, base_width * 2, base_width * 4, base_width * 8]
        self.incre_modules, self.downsamp_modules, self.final_layer = \
            self._make_head(block_1, pre_stage_channels, outplanes=2048, conv=stage_with_conv[3], ctx=stage_with_ctx[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return [64, self.base_width, self.base_width * 2, self.base_width * 4, self.base_width * 8]

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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, conv='Conv2d', ctx=''):
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
            block(self.inplanes, planes, 64, 1, stride, dilation, radix=self.radix, downsample=downsample,
                  stride_3x3=True, conv=conv, norm=self.norm, ctx=ctx)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, 64, 1, 1, dilation, radix=self.radix, downsample=None,
                      stride_3x3=True, conv=conv, norm=self.norm, ctx=ctx)
            )

        return nn.Sequential(*layers)

    def _make_transition(self, index=1, stride=1):
        transition = nn.ModuleList()
        if index == 1:
            transition.append(nn.Sequential(
                nn.Conv2d(self.inplanes, self.base_width, kernel_size=3, stride=1, padding=1, bias=False),
                make_norm(self.base_width, norm=self.norm.replace('Mix', '')),
                nn.ReLU(inplace=True),
            ))
        else:
            transition.extend([nn.Sequential() for _ in range(index)])
        transition.append(nn.Sequential(
            nn.Sequential(  # Double Sequential to fit with official pre-trained weights
                nn.Conv2d(self.inplanes if index == 1 else self.base_width * (2 ** (index - 1)),
                          self.base_width * (2 ** index), kernel_size=3, stride=stride, padding=1, bias=False),
                make_norm(self.base_width * (2 ** index), norm=self.norm.replace('Mix', '')),
                nn.ReLU(inplace=True),
            )
        ))

        return transition

    def _make_head(self, block, pre_stage_channels, outplanes=2048, conv='Conv2d', ctx=''):
        # Increasing the #channels on each resolution, from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            self.inplanes = channels
            incre_module = self._make_layer(block, self.head_dim[i], 1, stride=1, dilation=1, conv=conv, ctx=ctx)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = self.head_dim[i] * block.expansion
            out_channels = self.head_dim[i + 1] * block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),  # official implementation forgets bias=False
                make_norm(out_channels, norm=self.norm.replace('Mix', '')),
                nn.ReLU(inplace=True)
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(self.head_dim[3] * block.expansion, outplanes, 1, 1, 0),
            make_norm(outplanes, norm=self.norm.replace('Mix', '')),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        y = self.incre_modules[0](x[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](x[i + 1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)

        return y

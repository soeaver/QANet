import math

import torch
import torch.nn as nn

from lib.layers.wrappers import get_conv_op, get_norm_op, make_act, make_conv, make_ctx, make_norm
from lib.ops import SeConv2d, SplAtConv2d
from lib.utils.net import make_divisible


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, base_width=64, cardinality=1, stride=1, dilation=1, radix=1, downsample=None,
                 stride_3x3=False, conv='Conv2d', norm='BN', ctx=''):
        super(BasicBlock, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = make_conv(inplanes, width, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False, conv=conv)
        self.bn1 = make_norm(width, norm=norm, an_k=10 if planes < 256 else 20)
        self.conv2 = make_conv(width, width, kernel_size=3, stride=1, dilation=dilation, padding=dilation,
                               bias=False, conv=conv)
        self.bn2 = make_norm(width, norm=norm, an_k=10 if planes < 256 else 20)

        self.ctx = make_ctx(width, int(width * 0.0625), ctx=ctx)  # ctx_ratio=1 / 16.

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.ctx is not None:
            out = self.ctx(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, base_width=64, cardinality=1, stride=1, dilation=1, radix=1, downsample=None,
                 stride_3x3=False, conv='Conv2d', norm='BN', ctx=''):
        super(Bottleneck, self).__init__()
        (str1x1, str3x3) = (1, stride) if stride_3x3 else (stride, 1)
        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality
        self.radix = radix

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=str1x1, padding=0, bias=False)
        self.bn1 = make_norm(D * C, norm=norm.replace('Mix', ''))

        if radix > 1 and (str3x3 > 1 or dilation > 1):
            self.avd_layer = nn.AvgPool2d(3, str3x3, padding=1)
            str3x3 = 1
        else:
            self.avd_layer = None
        if radix > 1:
            self.conv2 = SplAtConv2d(
                D * C, D * C, kernel_size=3, stride=str3x3, padding=dilation, dilation=dilation, groups=C, bias=False,
                radix=radix, conv_op=get_conv_op(conv=conv), norm_op=get_norm_op(norm=norm)
            )
        else:
            self.conv2 = make_conv(
                D * C, D * C, kernel_size=3, stride=str3x3, padding=dilation, dilation=dilation, groups=C, bias=False,
                conv=conv
            )
            self.bn2 = make_norm(D * C, norm=norm, an_k=10 if planes < 256 else 20)

        self.conv3 = nn.Conv2d(D * C, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = make_norm(planes * self.expansion, norm=norm.replace('Mix', ''))

        self.ctx = make_ctx(planes * self.expansion, int(planes * self.expansion * 0.0625),
                            ctx=ctx)  # ctx_ratio=1 / 16.

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            out = self.relu(out)
        if self.avd_layer is not None:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.ctx is not None:
            out = self.ctx(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AlignedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, base_width=64, cardinality=1, stride=1, dilation=1, radix=1, downsample=None,
                 stride_3x3=False, conv='Conv2d', norm='BN', ctx=''):
        super(AlignedBottleneck, self).__init__()
        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality
        self.radix = radix

        if radix > 1 and (stride > 1 or dilation > 1):
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        else:
            self.avd_layer = None

        self.conv1_1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_1 = make_norm(D * C, norm=norm.replace('Mix', ''))

        if radix > 1:
            self.conv1_2 = SplAtConv2d(
                D * C, D * C, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=C, bias=False,
                radix=radix, conv_op=get_conv_op(conv=conv), norm_op=get_norm_op(norm=norm)
            )
        else:
            self.conv1_2 = make_conv(D * C, D * C, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                                     groups=C, bias=False, conv=conv)

        self.conv2_1 = nn.Conv2d(inplanes, D * C // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_1 = make_norm(D * C // 2, norm=norm.replace('Mix', ''))

        if radix > 1:
            self.conv2_2 = SplAtConv2d(
                D * C // 2, D * C // 2, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                groups=math.ceil(C / 2), bias=False, radix=radix, conv_op=get_conv_op(conv=conv),
                norm_op=get_norm_op(norm=norm)
            )
            self.conv2_3 = SplAtConv2d(
                D * C // 2, D * C // 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                groups=math.ceil(C / 2), bias=False, radix=radix, conv_op=get_conv_op(conv=conv),
                norm_op=get_norm_op(norm=norm)
            )
        else:
            self.conv2_2 = make_conv(D * C // 2, D * C // 2, kernel_size=3, stride=stride, padding=dilation,
                                     dilation=dilation, groups=math.ceil(C / 2), bias=False, conv=conv)
            self.bn2_2 = make_norm(D * C // 2, norm=norm, an_k=10 if planes < 256 else 20)
            self.conv2_3 = make_conv(D * C // 2, D * C // 2, kernel_size=3, stride=1, padding=dilation,
                                     dilation=dilation, groups=math.ceil(C / 2), bias=False, conv=conv)

        if radix == 1:
            self.bn_concat = make_norm(D * C + (D * C // 2), norm=norm, an_k=10 if planes < 256 else 20)

        self.conv = nn.Conv2d(D * C + (D * C // 2), planes * self.expansion, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.bn = make_norm(planes * self.expansion, norm=norm.replace('Mix', ''))

        self.ctx = make_ctx(planes * self.expansion, int(planes * self.expansion * 0.0625),
                            ctx=ctx)  # ctx_ratio=1 / 16.

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        branch1 = self.conv1_1(x)
        branch1 = self.bn1_1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.conv1_2(branch1)

        branch2 = self.conv2_1(x)
        branch2 = self.bn2_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_2(branch2)
        if self.radix == 1:
            branch2 = self.bn2_2(branch2)
            branch2 = self.relu(branch2)
        branch2 = self.conv2_3(branch2)

        out = torch.cat((branch1, branch2), 1)
        if self.radix == 1:
            out = self.bn_concat(out)
            out = self.relu(out)
        if self.avd_layer is not None:
            out = self.avd_layer(out)

        out = self.conv(out)
        out = self.bn(out)
        if self.ctx is not None:
            out = self.ctx(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inplanes, outplanes, kernel=3, stride=1, dilation=1, groups=(1, 1), t=6, norm='BN', bn_eps=1e-5,
                 act='ReLU6', se_ratio=0, **kwargs):
        super(InvertedResidual, self).__init__()
        inner_divisible = kwargs.pop("inner_divisible", False)
        force_residual = kwargs.pop("force_residual", False)
        se_reduce_mid = kwargs.pop("se_reduce_mid", False)
        se_divisible = kwargs.pop("se_divisible", False)
        se_out_act = kwargs.pop("se_out_act", 'Sigmoid')
        sync_se_act = kwargs.pop("sync_se_act", True)

        padding = (dilation * kernel - dilation) // 2
        self.stride = stride
        self.inplanes, self.outplanes = int(inplanes), int(outplanes)
        innerplanes = make_divisible(inplanes * t, 8) if inner_divisible else int(inplanes * abs(t))
        self.t = t
        self.force_residual = force_residual
        if self.t != 1:
            self.conv1 = nn.Conv2d(self.inplanes, innerplanes, kernel_size=1, padding=0, stride=1, groups=groups[0],
                                   bias=False)
            self.bn1 = make_norm(innerplanes, eps=bn_eps, norm=norm)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, padding=padding, stride=stride,
                               dilation=dilation, groups=innerplanes, bias=False)
        self.bn2 = make_norm(innerplanes, eps=bn_eps, norm=norm)
        se_base_chs = innerplanes if se_reduce_mid else self.inplanes
        se_innerplanse = make_divisible(se_base_chs * se_ratio, 8) if se_divisible else int(se_base_chs * se_ratio)
        if se_ratio:
            self.se = SeConv2d(
                innerplanes, se_innerplanse, inner_act=act if sync_se_act else 'ReLU', out_act=se_out_act
            )
        else:
            self.se = None
        self.conv3 = nn.Conv2d(innerplanes, self.outplanes, kernel_size=1, padding=0, stride=1, groups=groups[1],
                               bias=False)
        self.bn3 = make_norm(self.outplanes, eps=bn_eps, norm=norm)

        self.act = make_act(act=act)

    def forward(self, x):
        if self.stride == 1 and self.inplanes == self.outplanes and (self.t != 1 or self.force_residual):
            residual = x
        else:
            residual = None

        if self.t != 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)
        else:
            out = x
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        if self.se is not None:
            out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out if residual is None else out + residual

        return out

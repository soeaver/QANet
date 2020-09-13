import torch
import torch.nn.functional as F
from torch import nn


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 radix=2, reduction_factor=4, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, dropblock_prob=0.0):
        super(SplAtConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.out_channels = out_channels
        self.dropblock_prob = dropblock_prob
        self.conv = conv_op(in_channels, out_channels * radix, kernel_size, stride, padding, dilation,
                            groups=groups * radix, bias=bias)
        self.use_bn = norm_op is not None
        self.bn0 = norm_op(out_channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_op(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, out_channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, channel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

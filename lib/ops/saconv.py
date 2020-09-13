import torch
from torch import nn
from torch.nn import functional as F

from .conv2d_ws import Conv2dAWS
from .deform_conv import _DeformConv

deform_conv = _DeformConv.apply


class SAConv2d(Conv2dAWS):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 use_deform=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.use_deform = use_deform

        self.switch = torch.nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=stride, bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)

        self.weight_diff = nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()

        self.pre_context = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)

        self.post_context = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

        if self.use_deform:
            self.offset_s = nn.Conv2d(self.in_channels, 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=3,
                                      padding=1, stride=stride, bias=True)
            self.offset_l = nn.Conv2d(self.in_channels, 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=3,
                                      padding=1, stride=stride, bias=True)
            self.offset_s.weight.data.fill_(0)
            self.offset_s.bias.data.fill_(0)
            self.offset_l.weight.data.fill_(0)
            self.offset_l.bias.data.fill_(0)

    def forward(self, x):
        # pre-context
        avg_x = F.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x

        # switch
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)

        # sac
        weight = self._get_weight(self.weight)
        if self.use_deform:
            offset = self.offset_s(avg_x)
            out_s = deform_conv(x, offset, weight, self.stride, self.padding, self.dilation, self.groups, 1)
        else:
            out_s = super().conv2d_forward(x, weight)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        if self.use_deform:
            offset = self.offset_l(avg_x)
            out_l = deform_conv(x, offset, weight, self.stride, self.padding, self.dilation, self.groups, 1)
        else:
            out_l = super().conv2d_forward(x, weight)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d

        # post-context
        avg_x = F.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x

        return out


class CxConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CxConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                       dilation=dilation, groups=groups, bias=bias)

        self.pre_context = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)

        self.post_context = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

    def forward(self, x):
        # pre-context
        avg_x = F.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x

        # conv
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # post-context
        avg_x = F.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x

        return x

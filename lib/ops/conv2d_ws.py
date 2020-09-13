import torch
from torch import nn
from torch.nn import functional as F


class Conv2dWS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dAWS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dAWS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        self.register_buffer('weight_gamma', torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('weight_beta', torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = weight / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x):
        weight = self._get_weight(self.weight)
        return super().conv2d_forward(x, weight)

"""
Creates a RegNet Model as defined in:
Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Dollar, Piotr. (2020 CVPR).
Designing Network Design Spaces.
Copyright (c) Yang Lu, 2020
"""
import numpy as np

import torch.nn as nn

import lib.ops as ops
from lib.layers.wrappers import make_conv, make_norm


class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, bot_mul=1.0, group_width=1, stride=1, dilation=1, conv='Conv2d', norm='BN',
                 ctx=0.0):
        super(Bottleneck, self).__init__()
        innerplanes = int(round(outplanes * bot_mul))
        groups = innerplanes // group_width

        self.proj_block = (inplanes != outplanes) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
            self.bn = make_norm(outplanes, norm=norm)

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = make_norm(innerplanes, norm=norm)

        self.conv2 = make_conv(innerplanes, innerplanes, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, groups=groups, bias=False, conv=conv)
        self.bn2 = make_norm(innerplanes, norm=norm)
        if ctx:
            self.ctx = ops.SeConv2d(innerplanes, int(round(inplanes * ctx)))
        else:
            self.ctx = None
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = make_norm(outplanes, norm=norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.ctx is not None:
            out = self.ctx(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.proj_block:
            x = self.bn(self.proj(x))

        out += x
        out = self.relu(out)

        return out


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(nn.Module):
    def __init__(self, depth=25, w0=88, wa=26.31, wm=2.25, group_w=48, bot_mul=1.0, stride=2,
                 stage_with_conv=('Conv2d', 'Conv2d', 'Conv2d', 'Conv2d'), norm='BN', ctx=0.0,
                 num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(RegNet, self).__init__()
        block = Bottleneck
        stem_width = 32
        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(wa, w0, wm, depth)
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [group_w for _ in range(num_s)]
        bms = [bot_mul for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [stride for _ in range(num_s)]

        self.norm = norm
        self.ws = ws
        self.stem_width = stem_width

        self.inplanes = stem_width
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 2, 1, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=norm)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, ws[0], ss[0], ds[0], bms[0], gws[0], conv=stage_with_conv[0], ctx=ctx)
        self.layer2 = self._make_layer(block, ws[1], ss[1], ds[1], bms[1], gws[1], conv=stage_with_conv[1], ctx=ctx)
        self.layer3 = self._make_layer(block, ws[2], ss[2], ds[2], bms[2], gws[2], conv=stage_with_conv[2], ctx=ctx)
        self.layer4 = self._make_layer(block, ws[3], ss[3], ds[3], bms[3], gws[3], conv=stage_with_conv[2], ctx=ctx)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inplanes, num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return [self.stem_width, self.ws[0], self.ws[1], self.ws[2], self.ws[3]]

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
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, w, s, d, bm, gw, conv='Conv2d', ctx=0.0):
        layers = []
        for i in range(d):
            layers.append(
                block(self.inplanes, w, bm, gw, s if i == 0 else 1, dilation=1, conv=conv, norm=self.norm, ctx=ctx)
            )
            self.inplanes = w

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

import torch.nn as nn

import lib.ops as ops


def get_conv_op(conv="Conv2d", **kwargs):
    if len(conv) == 0:
        return None
    conv_op = {
        "Conv2d": nn.Conv2d,
        "Conv2dWS": ops.Conv2dWS,
        "DeformConv": ops.DeformConvPack,
        "MDeformConv": ops.ModulatedDeformConvPack,
    }[conv]

    return conv_op


def get_norm_op(norm="BN", eps=1e-5, group=32, an_k=10, **kwargs):
    if len(norm) == 0:
        return None
    norm_op = {
        "BN": lambda _num_channels: nn.BatchNorm2d(_num_channels, eps),
        "BN1d": lambda _num_channels: nn.BatchNorm1d(_num_channels, eps),
        "GN": lambda _num_channels: nn.GroupNorm(group, _num_channels, eps),
        "FrozenBN": lambda _num_channels: ops.FrozenBatchNorm2d(_num_channels, eps),
        "SyncBN": lambda _num_channels: ops.NaiveSyncBatchNorm(_num_channels, eps),
        "MixBN": lambda _num_channels: ops.MixtureBatchNorm2d(an_k, _num_channels, eps),
        "MixGN": lambda _num_channels: ops.MixtureGroupNorm(an_k, group, _num_channels, eps),
    }[norm]

    return norm_op


def make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=None,
              conv="Conv2d", **kwargs):
    norm = kwargs.pop("norm", None)
    act = kwargs.pop("act", None)
    _padding = (dilation * kernel_size - dilation) // 2 if padding is None else padding
    if bias is None:
        _bias = True if norm is None else False
    else:
        _bias = bias

    conv_op = get_conv_op(conv=conv)

    module = [conv_op(in_channels, out_channels, kernel_size, stride, _padding, dilation, groups, bias=_bias), ]
    if norm is not None:
        module.append(norm)
    if act is not None:
        module.append(act)

    if len(module) > 1:
        return nn.Sequential(*module)

    return module[0]


def make_fc(in_features, out_features, bias=True, **kwargs):
    norm = kwargs.pop("norm", None)
    act = kwargs.pop("act", None)

    fc = nn.Linear(in_features, out_features, bias=bias)

    module = [fc, ]
    if norm is not None:
        module.append(norm)
    if act is not None:
        module.append(act)

    if len(module) > 1:
        return nn.Sequential(*module)

    return module[0]


def make_norm(num_channels, eps=1e-5, norm='BN', **kwargs):
    group = kwargs.pop("group", 32)
    an_k = kwargs.pop("an_k", 10)

    if 'GN' in norm:
        assert num_channels % group == 0

    norm_op = get_norm_op(norm=norm, eps=eps, group=group, an_k=an_k)
    if norm_op is None:
        return norm_op

    return norm_op(num_channels)


def make_act(act='ReLU', **kwargs):
    inplace = kwargs.pop("inplace", True)

    if len(act) == 0:
        return None
    act = {
        "ReLU": nn.ReLU(inplace=inplace),
        "ReLU6": nn.ReLU6(inplace=inplace),
        "PReLU": nn.PReLU(),
        "LeakyReLU": nn.LeakyReLU(inplace=inplace),
        "H_Sigmoid": nn.Hardsigmoid(),
        "Sigmoid": nn.Sigmoid(),
        "TanH": nn.Tanh(),
        "H_Swish": nn.Hardswish(),
        "Swish": ops.Swish(),   # torch >= 1.7.0, nn.SiLU()
        "Mish": ops.Mish(),
    }[act]

    return act


def make_ctx(inplanes, innerplanse, ctx='', **kwargs):
    if len(ctx) == 0:
        return None
    ctx = {
        "SE": ops.SeConv2d,
        "GCB": ops.GlobalContextBlock,
        "ECA": ops.ECA,
    }[ctx]

    return ctx(inplanes, innerplanse)

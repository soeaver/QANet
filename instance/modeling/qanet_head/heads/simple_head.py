import torch
import torch.nn as nn

from instance.modeling import registry
from instance.core.config import cfg
from utils.net import make_conv


@registry.QANET_HEADS.register("simple_none_head")
class simple_none_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in[-1]

        self.dim_out = self.dim_in

    def forward(self, x):
        return x[-1]


@registry.QANET_HEADS.register("convx_head")
class convx_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in[-1]

        num_convs = 4
        conv_dim = 256
        use_lite = False
        use_bn = True
        use_gn = False
        use_dcn = False

        convx = []
        for i in range(num_convs):
            conv_type = 'deform' if use_dcn and i == num_convs - 1 else 'normal'
            convx.append(
                make_conv(self.dim_in, conv_dim, kernel=3, stride=1, dilation=1, use_dwconv=use_lite,
                          conv_type=conv_type, use_bn=use_bn, use_gn=use_gn, use_relu=True, kaiming_init=False,
                          suffix_1x1=use_lite)
            )
            self.dim_in = conv_dim
        self.add_module('convx', nn.Sequential(*convx))

        self.dim_out = self.dim_in

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convx(x[-1])
        return x

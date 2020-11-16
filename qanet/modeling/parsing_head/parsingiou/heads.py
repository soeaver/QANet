from torch import nn
from torch.nn import functional as F

from lib.layers import make_act, make_conv, make_fc, make_norm

from qanet.modeling import registry


@registry.PARSINGIOU_HEADS.register("parsingiou_head")
class ParsingIoUHead(nn.Module):
    """
    ParsingIoU head feature extractor.
    """

    def __init__(self, cfg, dim_in, spatial_in):
        super(ParsingIoUHead, self).__init__()

        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in[-1]

        num_convs = cfg.PARSING.PARSINGIOU.NUM_CONVS  # default = 2
        conv_dim = cfg.PARSING.PARSINGIOU.CONV_DIM
        norm = cfg.PARSING.PARSINGIOU.NORM

        self.conv1x1 = make_conv(self.dim_in, self.dim_in, kernel_size=1, stride=1,
                                 norm=make_norm(self.dim_in, norm=norm), act=make_act())

        conv_layers = []
        for i in range(num_convs):
            conv_layers.append(
                make_conv(self.dim_in, conv_dim, kernel_size=1, stride=1,
                          norm=make_norm(conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = conv_dim

        self.add_module('conv_layers', nn.Sequential(*conv_layers))

        self.dim_out = [conv_dim]
        self.spatial_out = [(1, 1), ]

        self._init_weights()

    def _init_weights(self):
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x[-1]
        x = self.conv1x1(x)
        x = self.conv_layers(F.adaptive_avg_pool2d(x, (1, 1)))

        return [x]

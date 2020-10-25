import torch
from torch import nn
from torch.nn import functional as F

from lib.layers import make_conv, make_norm, make_fc, make_act

from instance.modeling import registry


@registry.QUALITY_HEADS.register("quality_head")
class QualityHead(nn.Module):
    """
    Quality head for quality.
    """

    def __init__(self, cfg, dim_in, spatial_in):
        super(QualityHead, self).__init__()

        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in

        num_share_convs = cfg.PARSING.QUALITY.NUM_SHARE_CONVS
        num_parsing_convs = cfg.PARSING.QUALITY.NUM_PARSING_CONVS
        num_iou_convs = cfg.PARSING.QUALITY.NUM_IOU_CONVS
        parsing_conv_dim = cfg.PARSING.QUALITY.PARSING_CONV_DIM
        iou_conv_dim = cfg.PARSING.QUALITY.IOU_CONV_DIM
        norm = cfg.PARSING.QUALITY.NORM

        share_layers = []
        for i in range(num_share_convs):
            share_layers.append(
                make_conv(self.dim_in, parsing_conv_dim, kernel_size=3, stride=1, dilation=1,
                          norm=make_norm(parsing_conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = parsing_conv_dim

        parsing_layers = []
        for i in range(num_parsing_convs):
            parsing_layers.append(
                make_conv(self.dim_in, parsing_conv_dim, kernel_size=3, stride=1, dilation=1,
                          norm=make_norm(parsing_conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = parsing_conv_dim

        iou_layers = []
        for i in range(num_iou_convs):
            iou_layers.append(
                make_conv(self.dim_in, iou_conv_dim, kernel_size=1, stride=1,
                          norm=make_norm(iou_conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = iou_conv_dim

        self.add_module('share_layers', nn.Sequential(*share_layers))
        self.add_module('parsing_layers', nn.Sequential(*parsing_layers))
        self.add_module('iou_layers', nn.Sequential(*iou_layers))

        self.dim_out = [parsing_conv_dim, iou_conv_dim]
        self.spatial_out = [spatial_in[-1], (1, 1)]

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
        x = self.share_layers(x)
        xp = self.parsing_layers(x)
        xi = self.iou_layers(F.adaptive_avg_pool2d(x, (1, 1)))

        return [xp, xi]

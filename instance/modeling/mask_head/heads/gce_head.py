import torch
from torch import nn
from torch.nn import functional as F

from models.ops import NonLocal2d
from instance.modeling import registry
from instance.core.config import cfg
from utils.net import make_conv


@registry.MASK_HEADS.register("gce_head")
class gce_head(nn.Module):
    def __init__(self, dim_in):
        super(gce_head, self).__init__()
        self.dim_in = dim_in[-1]

        use_nl = cfg.MASK.GCE_HEAD.USE_NL
        use_bn = cfg.MASK.GCE_HEAD.USE_BN
        use_gn = cfg.MASK.GCE_HEAD.USE_GN
        conv_dim = cfg.MASK.GCE_HEAD.CONV_DIM
        asppv3_dim = cfg.MASK.GCE_HEAD.ASPPV3_DIM
        num_convs_before_asppv3 = cfg.MASK.GCE_HEAD.NUM_CONVS_BEFORE_ASPPV3
        asppv3_dilation = cfg.MASK.GCE_HEAD.ASPPV3_DILATION
        num_convs_after_asppv3 = cfg.MASK.GCE_HEAD.NUM_CONVS_AFTER_ASPPV3

        # convx before asppv3 module
        before_asppv3_list = []
        for _ in range(num_convs_before_asppv3):
            before_asppv3_list.append(
                make_conv(self.dim_in, conv_dim, kernel=3, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
            )
            self.dim_in = conv_dim
        self.conv_before_asppv3 = nn.Sequential(*before_asppv3_list) if len(before_asppv3_list) else None

        # asppv3 module
        self.asppv3 = []
        self.asppv3.append(
            make_conv(self.dim_in, asppv3_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
        )
        for dilation in asppv3_dilation:
            self.asppv3.append(
                make_conv(self.dim_in, asppv3_dim, kernel=3, dilation=dilation, use_bn=use_bn, use_gn=use_gn,
                          use_relu=True)
            )
        self.asppv3 = nn.ModuleList(self.asppv3)
        self.im_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            make_conv(self.dim_in, asppv3_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
        )
        self.dim_in = (len(asppv3_dilation) + 2) * asppv3_dim

        feat_list = []
        feat_list.append(
            make_conv(self.dim_in, conv_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
        )
        if use_nl:
            feat_list.append(
                NonLocal2d(conv_dim, int(conv_dim * cfg.MASK.GCE_HEAD.NL_RATIO), conv_dim, use_gn=True)
            )
        self.feat = nn.Sequential(*feat_list)
        self.dim_in = conv_dim

        # convx after asppv3 module
        assert num_convs_after_asppv3 >= 1
        after_asppv3_list = []
        for _ in range(num_convs_after_asppv3):
            after_asppv3_list.append(
                make_conv(self.dim_in, conv_dim, kernel=3, use_bn=use_bn, use_gn=use_gn, use_relu=True)
            )
            self.dim_in = conv_dim
        self.conv_after_asppv3 = nn.Sequential(*after_asppv3_list) if len(after_asppv3_list) else None
        self.dim_out = self.dim_in

    def forward(self, x):
        x_out = x[-1]
        input_size = x_out.size()

        if self.conv_before_asppv3 is not None:
            x_out = self.conv_before_asppv3(x_out)

        asppv3_out = [F.interpolate(self.im_pool(x_out), (input_size[2], input_size[3]),
                                    mode="bilinear", align_corners=False)]
        for i in range(len(self.asppv3)):
            asppv3_out.append(self.asppv3[i](x_out))
        asppv3_out = torch.cat(asppv3_out, 1)
        asppv3_out = self.feat(asppv3_out)

        if self.conv_after_asppv3 is not None:
            asppv3_out = self.conv_after_asppv3(asppv3_out)
        return asppv3_out

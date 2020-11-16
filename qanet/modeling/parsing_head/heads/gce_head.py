from torch import nn

from lib.layers import make_conv, make_norm, make_act, ASPP, NonLocal2d

from qanet.modeling import registry


@registry.PARSING_HEADS.register("gce_head")
class GCEHead(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(GCEHead, self).__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in

        use_nl = cfg.PARSING.GCE_HEAD.USE_NL
        norm = cfg.PARSING.GCE_HEAD.NORM
        conv_dim = cfg.PARSING.GCE_HEAD.CONV_DIM
        aspp_dim = cfg.PARSING.GCE_HEAD.ASPP_DIM
        num_convs_before_aspp = cfg.PARSING.GCE_HEAD.NUM_CONVS_BEFORE_ASPP
        aspp_dilation = cfg.PARSING.GCE_HEAD.ASPP_DILATION
        num_convs_after_aspp = cfg.PARSING.GCE_HEAD.NUM_CONVS_AFTER_ASPP

        # convx before aspp
        before_aspp_list = []
        for _ in range(num_convs_before_aspp):
            before_aspp_list.append(
                make_conv(self.dim_in, conv_dim, kernel_size=3, norm=make_norm(conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = conv_dim
        self.conv_before_aspp = nn.Sequential(*before_aspp_list) if len(before_aspp_list) else None

        # aspp
        self.aspp = ASPP(self.dim_in, aspp_dim, dilations=aspp_dilation, norm=norm)
        self.dim_in = self.aspp.dim_out

        feat_list = [
            make_conv(self.dim_in, conv_dim, kernel_size=1, norm=make_norm(conv_dim, norm=norm), act=make_act())
        ]
        # non-local
        if use_nl:
            feat_list.append(
                NonLocal2d(conv_dim, int(conv_dim * cfg.KRCNN.GCE_HEAD.NL_RATIO), conv_dim, use_gn=True)
            )
        self.feat = nn.Sequential(*feat_list)
        self.dim_in = conv_dim

        # convx after aspp
        assert num_convs_after_aspp >= 1
        after_aspp_list = []
        for _ in range(num_convs_after_aspp):
            after_aspp_list.append(
                make_conv(self.dim_in, conv_dim, kernel_size=3, norm=make_norm(conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = conv_dim
        self.conv_after_aspp = nn.Sequential(*after_aspp_list) if len(after_aspp_list) else None
        
        self.dim_out = [self.dim_in]
        self.spatial_out = [self.spatial_in]

        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x[-1]
        if self.conv_before_aspp is not None:
            x = self.conv_before_aspp(x)
        x = self.aspp(x)
        x = self.feat(x)
        if self.conv_after_aspp is not None:
            x = self.conv_after_aspp(x)
        return [x]

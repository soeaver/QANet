import torch.nn as nn

import lib.layers.fpn as fpn
from lib.utils.net import convert_conv2convws_model

from qanet.modeling import registry


# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
@registry.FPN_BODY.register("fpn")
class FPN(fpn.FPN):
    def __init__(self, cfg, dim_in, spatial_in):
        super(FPN, self).__init__()
        fpn_dim = cfg.FPN.DIM  # 256
        use_c5 = cfg.FPN.USE_C5
        norm = cfg.FPN.NORM
        min_level, max_level = cfg.FPN.LOWEST_BACKBONE_LVL, cfg.FPN.HIGHEST_BACKBONE_LVL    # 2, 5
        lowest_bk_lvl = cfg.FPN.LOWEST_BACKBONE_LVL  # 2
        highest_bk_lvl = cfg.FPN.HIGHEST_BACKBONE_LVL  # 5
        extra_conv = cfg.FPN.EXTRA_CONV_LEVELS

        self.dim_in = dim_in[-1]  # 2048
        self.spatial_in = spatial_in
        self.use_c5 = use_c5
        self.max_level = max_level
        self.lowest_bk_lvl = lowest_bk_lvl
        self.highest_bk_lvl = highest_bk_lvl
        self.extra_conv = extra_conv
        self.num_backbone_stages = len(dim_in) - (min_level - lowest_bk_lvl)  # 4

        self._make_layer(dim_in, fpn_dim, norm)

        output_levels = highest_bk_lvl - lowest_bk_lvl + 1
        # Retain only the spatial scales that will be used for RoI heads. `self.spatial_scale`
        # may include extra scales that are used for RPN proposals, but not for RoI heads.
        self.dim_out = [self.dim_in for _ in range(min_level - lowest_bk_lvl, output_levels)]
        self.spatial_out = self.spatial_in[min_level - lowest_bk_lvl:output_levels]

        if cfg.FPN.USE_WS:
            self = convert_conv2convws_model(self)

        self._init_weights()

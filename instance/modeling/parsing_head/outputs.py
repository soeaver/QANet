import torch
import torch.nn as nn
import torch.nn.functional as F

from instance.modeling import registry
from instance.core.config import cfg


@registry.PARSING_OUTPUTS.register("conv1x1_outputs")
class conv1x1_outputs(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.classify = nn.Conv2d(dim_in, cfg.PARSING.NUM_PARSING, kernel_size=1, stride=1, padding=0)
        if cfg.PARSING.PARSINGEDGE_ON:
            self.edge_parser = nn.Conv2d(dim_in, 2, kernel_size=1, stride=1, padding=0)
        self.spatial_scale = spatial_scale[0]

    def forward(self, x):
        parsing = self.classify(x)
        if cfg.PARSING.PARSINGEDGE_ON:
            edge = self.edge_parser(x)

        up_scale = int(1 / self.spatial_scale)
        if up_scale > 1:
            parsing = F.interpolate(parsing, scale_factor=up_scale, mode="bilinear", align_corners=False)
            if cfg.PARSING.PARSINGEDGE_ON:
                edge = F.interpolate(edge, scale_factor=up_scale, mode="bilinear", align_corners=False)
        if cfg.PARSING.PARSINGEDGE_ON:
            return [parsing, edge]
        else:
            return [parsing]

import torch.nn as nn

from instance.modeling import registry


@registry.PARSING_HEADS.register("simple_none_head")
class SimpleNoneHead(nn.Module):
    def __init__(self, cfg, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]

        self.dim_out = self.dim_in
        self.spatial_scale = spatial_scale

    def forward(self, x):
        return x[-1]

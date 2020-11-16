import torch.nn as nn

from qanet.modeling import registry


@registry.UV_HEADS.register("simple_none_head")
class SimpleNoneHead(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in[-1]

        self.dim_out = [self.dim_in]
        self.spatial_out = [self.spatial_in]

    def forward(self, x):
        return [x[-1]]

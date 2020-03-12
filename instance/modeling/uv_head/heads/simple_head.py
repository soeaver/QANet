import torch
import torch.nn as nn

from instance.modeling import registry
from instance.core.config import cfg


@registry.UV_HEADS.register("simple_none_head")
class simple_none_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in[-1]

        self.dim_out = self.dim_in

    def forward(self, x):
        return x[-1]

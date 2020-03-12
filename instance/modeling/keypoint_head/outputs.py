import torch
import torch.nn as nn

from instance.modeling import registry
from instance.core.config import cfg


@registry.KEYPOINT_OUTPUTS.register("conv1x1_outputs")
class conv1x1_outputs(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.classify = nn.Conv2d(dim_in, cfg.KEYPOINT.NUM_JOINTS, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.classify(x)
        return x
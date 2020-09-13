import torch.nn as nn
import torch.nn.functional as F

from instance.modeling import registry


@registry.PARSING_OUTPUTS.register("conv1x1_outputs")
class Conv1x1Outputs(nn.Module):
    def __init__(self, cfg, dim_in, spatial_scale):
        super().__init__()
        self.classify = nn.Conv2d(dim_in, cfg.PARSING.NUM_PARSING, kernel_size=1, stride=1, padding=0)
        self.spatial_scale = spatial_scale[0]

    def forward(self, x):
        x = self.classify(x)
        up_scale = int(1 / self.spatial_scale)
        if up_scale > 1:
            x = F.interpolate(x, scale_factor=up_scale, mode="bilinear", align_corners=False)
        return x

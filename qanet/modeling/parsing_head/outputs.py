import torch.nn as nn
import torch.nn.functional as F

from qanet.modeling import registry


@registry.PARSING_OUTPUTS.register("conv1x1_outputs")
class Conv1x1Outputs(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in

        self.classify = nn.Conv2d(self.dim_in, cfg.PARSING.NUM_PARSING, kernel_size=1, stride=1, padding=0)

        self.dim_out = [cfg.PARSING.NUM_PARSING]
        self.spatial_out = [1.0]

    def forward(self, x):
        x = x[-1]
        x = self.classify(x)
        up_scale = int(1 / self.spatial_in[0])
        if up_scale > 1:
            x = F.interpolate(x, scale_factor=up_scale, mode="bilinear", align_corners=False)
        return [x]

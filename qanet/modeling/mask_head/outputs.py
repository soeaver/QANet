import torch
import torch.nn as nn
import torch.nn.functional as F

from qanet.modeling import registry


@registry.MASK_OUTPUTS.register("conv1x1_outputs")
class Conv1x1Outputs(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in

        self.classify = nn.Conv2d(self.dim_in, cfg.MASK.NUM_CLASSES, kernel_size=1, stride=1, padding=0)
        
        self.dim_out = [cfg.MASK.NUM_CLASSES]
        self.spatial_out = [1.0]

    def forward(self, x, labels=None):
        x = x[-1]
        if labels is None:
            labels = torch.zeros(x.shape[0]).long()
        x = self.classify(x)
        x = x[range(len(labels)), labels].unsqueeze(1)
        up_scale = int(1 / self.spatial_in[0])
        if up_scale > 1:
            x = F.interpolate(
                x, scale_factor=up_scale, mode="bilinear", align_corners=False
            )
        return [x]

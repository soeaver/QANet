import torch.nn as nn

from qanet.modeling import registry


@registry.KEYPOINT_OUTPUTS.register("conv1x1_outputs")
class Conv1x1Outputs(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in
        self.classify = nn.Conv2d(self.dim_in, cfg.KEYPOINT.NUM_KEYPOINTS, kernel_size=1, stride=1, padding=0)
        self.dim_out = [cfg.KEYPOINT.NUM_KEYPOINTS]
        self.spatial_out = [self.spatial_in[0]]

    def forward(self, x):
        x = x[-1]
        x = self.classify(x)
        return [x]

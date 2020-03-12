import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ops import ConvTranspose2d
from instance.modeling import registry
from instance.core.config import cfg


@registry.UV_OUTPUTS.register("UV_outputs")
class UV_outputs(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.spatial_scale = spatial_scale[0]
        self.deconv_Ann = ConvTranspose2d(
            dim_in,
            15,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.deconv_Index = ConvTranspose2d(
            dim_in,
            25,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.deconv_U = ConvTranspose2d(
            dim_in,
            25,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.deconv_V = ConvTranspose2d(
            dim_in,
            25,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_Ann = self.deconv_Ann(x)
        x_Index = self.deconv_Index(x)
        x_U = self.deconv_U(x)
        x_V = self.deconv_V(x)
        up_scale = int(0.5 / self.spatial_scale)
        if up_scale > 1:
            x_Ann = F.interpolate(x_Ann, scale_factor=up_scale, mode="bilinear", align_corners=False)
            x_Index = F.interpolate(x_Index, scale_factor=up_scale, mode="bilinear", align_corners=False)
            x_U = F.interpolate(x_U, scale_factor=up_scale, mode="bilinear", align_corners=False)
            x_V = F.interpolate(x_V, scale_factor=up_scale, mode="bilinear", align_corners=False)
        if not self.training:
            # x_Index = F.softmax(x_Index, dim=1)
            x_Index = F.sigmoid(x_Index)
        return [x_Ann, x_Index, x_U, x_V]

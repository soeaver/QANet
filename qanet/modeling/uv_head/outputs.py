import torch.nn as nn
import torch.nn.functional as F

from qanet.modeling import registry


@registry.UV_OUTPUTS.register("UV_outputs")
class UVOutputs(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in
        num_parts = cfg.UV.NUM_PARTS
        num_patches = cfg.UV.NUM_PATCHES

        self.deconv_Ann = nn.ConvTranspose2d(self.dim_in, num_parts + 1, kernel_size=4, stride=2, padding=1)
        self.deconv_Index = nn.ConvTranspose2d(self.dim_in, num_patches + 1, kernel_size=4, stride=2, padding=1)
        self.deconv_U = nn.ConvTranspose2d(self.dim_in, num_patches + 1, kernel_size=4, stride=2, padding=1)
        self.deconv_V = nn.ConvTranspose2d(self.dim_in, num_patches + 1, kernel_size=4, stride=2, padding=1)

        self.dim_out = [num_parts + 1, num_patches + 1, num_patches + 1, num_patches + 1]
        self.spatial_out = [1.0, 1.0, 1.0, 1.0]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x[-1]
        x_Ann = self.deconv_Ann(x)
        x_Index = self.deconv_Index(x)
        x_U = self.deconv_U(x)
        x_V = self.deconv_V(x)
        up_scale = int(0.5 / self.spatial_in[0])
        if up_scale > 1:
            x_Ann = F.interpolate(x_Ann, scale_factor=up_scale, mode="bilinear", align_corners=False)
            x_Index = F.interpolate(x_Index, scale_factor=up_scale, mode="bilinear", align_corners=False)
            x_U = F.interpolate(x_U, scale_factor=up_scale, mode="bilinear", align_corners=False)
            x_V = F.interpolate(x_V, scale_factor=up_scale, mode="bilinear", align_corners=False)
        if not self.training:
            x_Ann = F.softmax(x_Ann, dim=1)
            x_Index = F.sigmoid(x_Index)

        return [x_Ann, x_Index, x_U, x_V]

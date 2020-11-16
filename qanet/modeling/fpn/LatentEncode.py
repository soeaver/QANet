import torch
import torch.nn as nn

from lib.layers import make_conv, make_norm

from qanet.modeling import registry


@registry.FPN_BODY.register("latent_encode")
class LatentEncode(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        self.dim_in = dim_in[-1]  # 2048
        self.spatial_in = spatial_in

        latent_dim = cfg.FPN.LATENC.CONV_DIM  # 256
        norm = cfg.FPN.LATENC.NORM
        self.lateral_convs = nn.ModuleList()
        for dim in dim_in:
            self.lateral_convs.append(
                nn.Sequential(
                    make_conv(dim, latent_dim, kernel_size=1, norm=make_norm(latent_dim, norm=norm)),
                    nn.AdaptiveAvgPool2d(1),
                    make_conv(latent_dim, latent_dim, kernel_size=1, norm=make_norm(latent_dim, norm=norm))
                )
            )

        self.encoder = nn.Sequential(
            nn.Conv2d(latent_dim * len(dim_in), latent_dim, 1),
            nn.Sigmoid()
        )

        self.dim_out = [latent_dim]
        self.spatial_out = [-1]

        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        latent_output_blobs = []
        for idx, layer in enumerate(self.lateral_convs):
            latent_output_blobs.append(layer(x[idx]))

        latent_feat = torch.cat(latent_output_blobs, 1)
        latent_feat = self.encoder(latent_feat)

        return latent_feat

import torch
import torch.nn as nn

from lib.layers import make_norm

from qanet.modeling import registry


@registry.FPN_BODY.register("xdeconv")
class XDeConv(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super().__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in[-1]

        hidden_dim = cfg.FPN.DECONVX.HEAD_DIM  # default: 256
        head_decay_factor = cfg.FPN.DECONVX.HEAD_DECAY_FACTOR  # default: 1
        self.deconv_kernel = cfg.FPN.DECONVX.HEAD_KERNEL  # default: 4
        padding, output_padding = self._get_deconv_param()
        deconv_with_bias = cfg.FPN.DECONVX.WITH_BIAS
        num_deconvs = cfg.FPN.DECONVX.NUM_DECONVS
        norm = cfg.FPN.DECONVX.NORM

        # deconv module
        deconv_list = []
        for _ in range(num_deconvs):
            deconv_list.extend([
                nn.ConvTranspose2d(self.dim_in, hidden_dim, kernel_size=self.deconv_kernel, stride=2,
                                   padding=padding, output_padding=output_padding, bias=deconv_with_bias),
                make_norm(hidden_dim, norm=norm),
                nn.ReLU(inplace=True)
            ])
            self.dim_in = hidden_dim
            hidden_dim //= head_decay_factor
            self.spatial_in *= 2
        self.deconv_module = nn.Sequential(*deconv_list)

        self.dim_out = [self.dim_in]
        self.spatial_out = [self.spatial_in]

        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_deconv_param(self):
        if self.deconv_kernel == 4:
            return 1, 0
        elif self.deconv_kernel == 3:
            return 1, 1
        elif self.deconv_kernel == 2:
            return 0, 0
        else:
            raise ValueError('only support POSE.SIMPLE.DECONV_HEAD_KERNEL in [2, 3, 4]')

    def forward(self, x):
        c5_out = x[-1]

        out = self.deconv_module(c5_out)

        return [out]

import torch
from torch import nn
from torch.nn import functional as F

from lib.layers import make_conv, make_norm, make_fc, make_act
from instance.modeling import registry


@registry.MASKIOU_HEADS.register("convx_head")
class MaskIoU_head(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(self, cfg, dim_in, spatial_scale):
        super(MaskIoU_head, self).__init__()

        self.spatial_scale = spatial_scale[0]
        self.dim_in = dim_in + 1
        num_stacked_convs = cfg.MASK.MASKIOU.NUM_STACKED_CONVS  # default = 2
        conv_dim = cfg.MASK.MASKIOU.CONV_DIM
        mlp_dim = cfg.MASK.MASKIOU.MLP_DIM
        norm = cfg.MASK.MASKIOU.NORM

        convx = []
        for _ in range(num_stacked_convs):
            layer_stride = 2 if _ == 0 else 1
            convx.append(
                make_conv(self.dim_in, conv_dim, kernel_size=3, stride=layer_stride,
                          norm=make_norm(conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = conv_dim
        self.convx = nn.Sequential(*convx)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maskiou_fc1 = make_fc(self.dim_in, mlp_dim)
        self.maskiou_fc2 = make_fc(mlp_dim, mlp_dim)
        self.dim_out = mlp_dim

        self._init_weights()

    def _init_weights(self):
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask_logits):
        down_scale = int(1 / self.spatial_scale)
        mask_pool = F.max_pool2d(mask_logits, kernel_size=down_scale, stride=down_scale)
        x = torch.cat((x, mask_pool), 1)
        x = self.convx(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))

        return x

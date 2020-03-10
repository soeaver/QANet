import torch
from torch import nn
from torch.nn import functional as F

from pet.utils.net import make_conv, make_fc
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


@registry.PARSINGIOU_HEADS.register("convx_head")
class ParsingIoU_head(nn.Module):
    """
    ParsingIoU head feature extractor.
    """

    def __init__(self, dim_in, spatial_scale):
        super(ParsingIoU_head, self).__init__()

        self.spatial_scale = spatial_scale[0]
        self.dim_in = dim_in + cfg.PARSING.NUM_PARSING
        num_stacked_convs = cfg.PARSING.PARSINGIOU.NUM_STACKED_CONVS  # default = 2
        conv_dim = cfg.PARSING.PARSINGIOU.CONV_DIM
        mlp_dim = cfg.PARSING.PARSINGIOU.MLP_DIM
        use_bn = cfg.PARSING.PARSINGIOU.USE_BN
        use_gn = cfg.PARSING.PARSINGIOU.USE_GN

        convx = []
        for _ in range(num_stacked_convs):
            layer_stride = 2 if _ == 0 else 1
            convx.append(
                make_conv(
                    self.dim_in, conv_dim, kernel=3, stride=layer_stride, use_bn=use_bn, use_gn=use_gn, use_relu=True
                )
            )
            self.dim_in = conv_dim
        self.convx = nn.Sequential(*convx)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.parsingiou_fc1 = make_fc(self.dim_in, mlp_dim, use_bn=False, use_gn=False)
        self.parsingiou_fc2 = make_fc(mlp_dim, mlp_dim, use_bn=False, use_gn=False)
        self.dim_out = mlp_dim

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

    def forward(self, x, parsing_logits):
        down_scale = int(1 / self.spatial_scale)
        parsing_pool = F.max_pool2d(parsing_logits, kernel_size=down_scale, stride=down_scale)
        x = torch.cat((x, parsing_pool), 1)
        x = self.convx(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.parsingiou_fc1(x))
        x = F.relu(self.parsingiou_fc2(x))

        return x

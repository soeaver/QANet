from torch import nn

from qanet.modeling import registry


@registry.PARSINGIOU_OUTPUTS.register("parsingiou_output")
class ParsingIoUOutput(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(ParsingIoUOutput, self).__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in

        num_classes = 1
        self.parsing_iou = nn.Linear(self.dim_in, num_classes)

        self.dim_out = [num_classes]
        self.spatial_out = [(1, 1), ]

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

    def forward(self, x):
        x = x[-1]
        x = self.parsing_iou(x.view(x.size(0), -1))
        return [x]

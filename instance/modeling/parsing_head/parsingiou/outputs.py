from torch import nn

from instance.modeling import registry


@registry.PARSINGIOU_OUTPUTS.register("parsingiou_output")
class ParsingIoUOutput(nn.Module):
    def __init__(self, cfg, dim_in):
        super(ParsingIoUOutput, self).__init__()

        self.num_parsing = cfg.PARSING.NUM_PARSING
        self.use_cla_iou = cfg.PARSING.PARSINGIOU.USE_CLA_IOU

        dim_out = self.num_parsing if self.use_cla_iou else 1
        self.parsing_iou = nn.Linear(dim_in, dim_out)

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
        x = self.parsing_iou(x.view(x.size(0), -1))
        return x

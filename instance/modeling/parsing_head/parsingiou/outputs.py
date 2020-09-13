from torch import nn

from instance.modeling import registry


@registry.PARSINGIOU_OUTPUTS.register("linear_output")
class ParsingIoUOutput(nn.Module):
    def __init__(self, cfg, dim_in):
        super(ParsingIoUOutput, self).__init__()
        num_classes = 1

        self.parsingiou = nn.Linear(dim_in, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.parsingiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.parsingiou.bias, 0)

    def forward(self, x):
        parsingiou = self.parsingiou(x)
        return parsingiou

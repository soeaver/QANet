import torch
from torch import nn

from instance.modeling import registry


@registry.MASKIOU_OUTPUTS.register("linear_output")
class MaskIoUOutput(nn.Module):
    def __init__(self, cfg, dim_in):
        super(MaskIoUOutput, self).__init__()
        num_classes = cfg.MASK.NUM_CLASSES

        self.maskiou = nn.Linear(dim_in, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)

    def forward(self, x, labels=None):
        if labels is None:
            labels = torch.zeros(x.shape[0]).long()
        x = self.maskiou(x)
        x = x[range(len(labels)), labels]
        return x

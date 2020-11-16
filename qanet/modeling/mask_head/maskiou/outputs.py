import torch
from torch import nn

from qanet.modeling import registry


@registry.MASKIOU_OUTPUTS.register("maskiou_output")
class MaskIoUOutput(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(MaskIoUOutput, self).__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in

        num_classes = cfg.MASK.NUM_CLASSES
        self.mask_iou = nn.Linear(self.dim_in, num_classes)

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

    def forward(self, x, labels=None):
        x = x[-1]
        if labels is None:
            labels = torch.zeros(x.shape[0]).long()
        x = self.mask_iou(x.view(x.size(0), -1))
        x = x[range(len(labels)), labels]
        return [x]

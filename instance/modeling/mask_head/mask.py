import torch

from instance.modeling.mask_head import heads
from instance.modeling.mask_head import outputs
from instance.modeling.mask_head.loss import mask_loss
from instance.modeling import registry
from instance.core.config import cfg


class Mask(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(Mask, self).__init__()
        head = registry.MASK_HEADS[cfg.MASK.MASK_HEAD]
        self.Head = head(dim_in)
        output = registry.MASK_OUTPUTS[cfg.MASK.MASK_OUTPUT]
        self.Output = output(self.Head.dim_out, spatial_scale)

        self.loss_evaluator = mask_loss

    def forward(self, conv_features, targets=None):
        mask_feat = self.Head(conv_features)
        output = self.Output(mask_feat)

        if not self.training:
            return output, {}

        device = torch.device(cfg.DEVICE)
        mask_targets = targets['mask'].to(device)
        lables = targets['mask_class'].to(device)
        loss_mask = self.loss_evaluator(output, mask_targets, lables)

        return None, dict(loss_mask=loss_mask)

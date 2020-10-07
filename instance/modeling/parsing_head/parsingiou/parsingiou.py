import torch
from torch import nn

from lib.datasets.structures.bounding_box import BoxList
from instance.modeling.parsing_head.parsingiou import heads
from instance.modeling.parsing_head.parsingiou import outputs
from instance.modeling.parsing_head.parsingiou.loss import parsingiou_loss_evaluator
from instance.modeling import registry


class ParsingIoU(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_scale):
        super(ParsingIoU, self).__init__()
        self.spatial_scale = spatial_scale

        head = registry.PARSINGIOU_HEADS[cfg.PARSING.PARSINGIOU.PARSINGIOU_HEAD]
        self.Head = head(cfg, dim_in, self.spatial_scale)
        output = registry.PARSINGIOU_OUTPUTS[cfg.PARSING.PARSINGIOU.PARSINGIOU_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out)

        self.loss_evaluator = parsingiou_loss_evaluator(cfg)

    def forward(self, features, parsingiou_targets=None):
        """
        Arguments:
            features (Tensor): feature-maps from possibly several levels
            parsingiou_targets (Tensor, optional): the ground-truth parsingiou targets.

        Returns:
            losses (Tensor): During training, returns the losses for the
                head. During testing, returns an empty dict.
            parsingiou_pre (Tensor): during training, returns None. During testing, the predicted parsingiou.
        """
        x = self.Head(features)
        parsingiou_pred = self.Output(x)

        if self.training:
            return self._forward_train(parsingiou_pred, parsingiou_targets)
        else:
            return self._forward_test(parsingiou_pred)

    def _forward_train(self, parsingiou_pred, parsingiou_targets=None):
        loss_parsingiou = self.loss_evaluator(parsingiou_pred, parsingiou_targets)
        return loss_parsingiou, None

    def _forward_test(self, parsingiou_pred):
        return {}, parsingiou_pred

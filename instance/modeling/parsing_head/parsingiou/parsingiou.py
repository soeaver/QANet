import torch
from torch import nn

from pet.utils.misc import logging_rank
from pet.utils.data.structures.bounding_box import BoxList
from pet.instance.modeling.parsing_head.parsingiou import heads
from pet.instance.modeling.parsing_head.parsingiou import outputs
from pet.instance.modeling.parsing_head.parsingiou.loss import parsingiou_loss_evaluator
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class ParsingIoU(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(ParsingIoU, self).__init__()
        head = registry.PARSINGIOU_HEADS[cfg.PARSING.PARSINGIOU.PARSINGIOU_HEAD]
        self.Head = head(dim_in, spatial_scale)
        output = registry.PARSINGIOU_OUTPUTS[cfg.PARSING.PARSINGIOU.PARSINGIOU_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.loss_evaluator = parsingiou_loss_evaluator()

    def forward(self, features, parsing_logits, parsingiou_targets=None):
        """
        Arguments:
            features (Tensor): feature-maps from possibly several levels
            parsing_logits (Tensor): targeted parsing
            parsingiou_targets (Tensor, optional): the ground-truth parsingiou targets.

        Returns:
            losses (Tensor): During training, returns the losses for the
                head. During testing, returns an empty dict.
            parsingiou (Tensor): during training, returns None. During testing, the predicted parsingiou.
        """
        x = self.Head(features, parsing_logits)
        pred_parsingiou = self.Output(x)

        if self.training:
            return self._forward_train(pred_parsingiou, parsingiou_targets)
        else:
            return self._forward_test(pred_parsingiou)

    def _forward_train(self, pred_parsingiou, parsingiou_targets=None):
        loss_parsingiou = self.loss_evaluator(pred_parsingiou, parsingiou_targets)
        return loss_parsingiou, None

    def _forward_test(self, pred_parsingiou):
        return {}, pred_parsingiou

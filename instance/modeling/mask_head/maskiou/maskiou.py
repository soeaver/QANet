import torch
from torch import nn

from lib.datasets.structures.bounding_box import BoxList
from instance.modeling.mask_head.maskiou import heads
from instance.modeling.mask_head.maskiou import outputs
from instance.modeling.mask_head.maskiou.loss import maskiou_loss_evaluator
from instance.modeling import registry


class MaskIoU(nn.Module):
    def __init__(self, cfg, dim_in, spatial_scale):
        super(MaskIoU, self).__init__()
        self.spatial_scale = spatial_scale

        head = registry.MASKIOU_HEADS[cfg.MASK.MASKIOU.MASKIOU_HEAD]
        self.Head = head(cfg, dim_in, self.spatial_scale)
        output = registry.MASKIOU_OUTPUTS[cfg.MASK.MASKIOU.MASKIOU_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out)

        self.loss_evaluator = maskiou_loss_evaluator(cfg)

    def forward(self, features, mask_logits, labels=None, maskiou_targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            mask_logits (list[Tensor]): mask logits
            labels (list[Tensor]): class label of mask
            maskiou_targets (list[Tensor], optional): the ground-truth maskiou targets.

        Returns:
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
            maskiou_pred (Tensor): during training, returns None. During testing, the predicted maskiou
        """
        x = self.Head(features, mask_logits)
        maskiou_pred = self.Output(x, labels)

        if self.training:
            return self._forward_train(maskiou_pred, maskiou_targets)
        else:
            return self._forward_test(maskiou_pred)

    def _forward_train(self, maskiou_pred, maskiou_targets=None):
        loss_maskiou = self.loss_evaluator(maskiou_pred, maskiou_targets)
        return loss_maskiou, None

    def _forward_test(self, maskiou_pred):
        return {}, maskiou_pred

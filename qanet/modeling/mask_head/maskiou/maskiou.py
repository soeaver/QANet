import torch
from torch import nn

from qanet.modeling import registry
from qanet.modeling.mask_head.maskiou import heads, outputs
from qanet.modeling.mask_head.maskiou.loss import maskiou_loss_evaluator


class MaskIoU(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(MaskIoU, self).__init__()
        self.dim_in = dim_in
        self.spatial_in = spatial_in

        head = registry.MASKIOU_HEADS[cfg.MASK.MASKIOU.MASKIOU_HEAD]
        self.Head = head(cfg, self.dim_in, self.spatial_in)
        output = registry.MASKIOU_OUTPUTS[cfg.MASK.MASKIOU.MASKIOU_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.loss_evaluator = maskiou_loss_evaluator(cfg)

        self.dim_out = self.Output.dim_out
        self.spatial_out = self.Output.spatial_out

    def forward(self, features, labels=None, maskiou_targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            labels (list[Tensor]): class label of mask
            maskiou_targets (list[Tensor], optional): the ground-truth maskiou targets.

        Returns:
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
            maskiou_pred (Tensor): during training, returns None. During testing, the predicted maskiou
        """
        x = self.Head(features)
        maskiou_pred = self.Output(x, labels)

        if self.training:
            return self._forward_train(maskiou_pred, maskiou_targets)
        else:
            return self._forward_test(maskiou_pred)

    def _forward_train(self, maskiou_pred, maskiou_targets=None):
        loss_maskiou = self.loss_evaluator(maskiou_pred, maskiou_targets)
        return loss_maskiou, None

    def _forward_test(self, maskiou_pred):
        return {}, maskiou_pred[-1]

import torch
from torch.nn import functional as F


class MaskLossComputation(object):
    def __init__(self, cfg):
        self.maskiou_on = cfg.MASK.MASKIOU_ON
        self.loss_weight = cfg.MASK.LOSS_WEIGHT

    def __call__(self, logits, mask_targets):
        mask_logits = logits[-1]
        mask_logits = mask_logits.squeeze(1)

        if self.maskiou_on:
            pred_masks = mask_logits.clone().detach()
            pred_masks[:] = pred_masks.sigmoid() > 0.5
            mask_ovr = pred_masks * mask_targets
            mask_ovr_area = mask_ovr.sum(dim=[1, 2])
            mask_union_area = pred_masks.sum(dim=[1, 2]) + mask_targets.sum(dim=[1, 2]) - mask_ovr_area
            value_1 = torch.ones(pred_masks.shape[0], device=mask_logits.device)
            value_0 = torch.zeros(pred_masks.shape[0], device=mask_logits.device)
            mask_union_area = torch.max(mask_union_area, value_1)
            mask_ovr_area = torch.max(mask_ovr_area, value_0)
            maskiou_targets = mask_ovr_area / mask_union_area
        else:
            maskiou_targets = None

        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_targets, reduction='mean')
        mask_loss *= self.loss_weight

        return mask_loss, maskiou_targets


def mask_loss_evaluator(cfg):
    loss_evaluator = MaskLossComputation(cfg)
    return loss_evaluator

import torch

from lib.ops import l2_loss


class MaskIoULossComputation(object):
    def __init__(self, cfg):
        self.loss_weight = cfg.MASK.MASKIOU.LOSS_WEIGHT

    def __call__(self, maskiou_pred, maskiou_gt):
        maskiou_pred = maskiou_pred[-1]
        maskiou_gt = maskiou_gt.detach()
        maskiou_loss = l2_loss(maskiou_pred, maskiou_gt)
        maskiou_loss = self.loss_weight * maskiou_loss

        return maskiou_loss


def maskiou_loss_evaluator(cfg):
    loss_evaluator = MaskIoULossComputation(cfg)
    return loss_evaluator

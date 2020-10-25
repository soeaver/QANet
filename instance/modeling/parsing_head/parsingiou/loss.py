import torch

from lib.ops import l2_loss


class ParsingIoULossComputation(object):
    def __init__(self, cfg):
        self.loss_weight = cfg.PARSING.PARSINGIOU.LOSS_WEIGHT

    def __call__(self, parsingiou_pred, parsingiou_gt):
        parsingiou_pred = parsingiou_pred[-1]
        parsingiou_gt = parsingiou_gt.detach()
        parsingiou_loss = l2_loss(parsingiou_pred[:, 0], parsingiou_gt)
        parsingiou_loss = self.loss_weight * parsingiou_loss

        return parsingiou_loss


def parsingiou_loss_evaluator(cfg):
    loss_evaluator = ParsingIoULossComputation(cfg)
    return loss_evaluator

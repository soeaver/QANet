import torch

from lib.ops import JointsMSELoss


class KeypointLossComputation(object):
    def __init__(self, cfg):
        self.loss_weight = cfg.KEYPOINT.LOSS_WEIGHT

    def __call__(self, logits, targets, target_weight=None):
        kpt_logits = logits[-1]
        device = kpt_logits.device
        criterion = JointsMSELoss().to(device)
        loss = criterion(kpt_logits, targets, target_weight)
        loss *= self.loss_weight

        return loss


def keypoint_loss_evaluator(cfg):
    loss_evaluator = KeypointLossComputation(cfg)
    return loss_evaluator

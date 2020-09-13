import torch

from instance.ops import JointsMSELoss


class KeypointLossComputation(object):
    def __init__(self, cfg):
        self.loss_weight = cfg.KEYPOINT.LOSS_WEIGHT

    def __call__(self, outputs, targets, target_weight=None):
        device = outputs.device
        criterion = JointsMSELoss().to(device)
        loss = criterion(outputs, targets, target_weight)
        loss *= self.loss_weight

        return loss


def keypoint_loss_evaluator(cfg):
    loss_evaluator = KeypointLossComputation(cfg)
    return loss_evaluator

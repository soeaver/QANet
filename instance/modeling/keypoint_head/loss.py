import torch

from pet.instance.ops import JointsMSELoss
from pet.instance.core.config import cfg


def keypoint_loss(outputs, targets, target_weight=None):
    device = torch.device(cfg.DEVICE)
    criterion = JointsMSELoss().to(device)
    loss = criterion(outputs, targets, target_weight)
    loss *= cfg.KEYPOINT.LOSS_WEIGHT

    return loss

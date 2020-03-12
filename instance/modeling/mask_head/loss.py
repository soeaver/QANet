import torch
from torch.nn import functional as F

from instance.core.config import cfg


def mask_loss(outputs, targets, labels):
    positive_inds = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[positive_inds]

    loss = F.binary_cross_entropy_with_logits(
        outputs[positive_inds, labels_pos], targets
    )
    loss *= cfg.MASK.LOSS_WEIGHT
    return loss

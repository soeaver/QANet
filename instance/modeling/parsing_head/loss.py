import numpy as np

import torch
from torch.nn import functional as F

from pet.instance.core.config import cfg


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_one_mean_iou(image_array, label_array, num_parsing):
    hist = fast_hist(label_array, image_array, num_parsing).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu


class ParsingLossComputation(object):
    def __init__(self, device, parsingiou_on):
        self.device = torch.device(device)
        self.parsingiou_on = parsingiou_on

    def __call__(self, parsing_logits, parsing_targets):
        if self.parsingiou_on:
            pred_parsings_np = parsing_logits.detach().argmax(dim=1).cpu().numpy()
            parsing_targets_np = parsing_targets.cpu().numpy()

            N = parsing_targets_np.shape[0]
            parsingiou_targets = np.zeros(N, dtype=np.float)

            for _ in range(N):
                parsing_iou = cal_one_mean_iou(parsing_targets_np[_], pred_parsings_np[_], cfg.PARSING.NUM_PARSING)
                parsingiou_targets[_] = np.nanmean(parsing_iou)
            parsingiou_targets = torch.from_numpy(parsingiou_targets).to(self.device, dtype=torch.float)

        parsing_targets = parsing_targets.to(self.device)
        parsing_loss = F.cross_entropy(parsing_logits, parsing_targets)
        parsing_loss *= cfg.PARSING.LOSS_WEIGHT

        if not self.parsingiou_on:
            return parsing_loss
        else:
            return parsing_loss, parsingiou_targets


def parsing_loss_evaluator():
    loss_evaluator = ParsingLossComputation(
        cfg.DEVICE, cfg.PARSING.PARSINGIOU_ON
    )
    return loss_evaluator

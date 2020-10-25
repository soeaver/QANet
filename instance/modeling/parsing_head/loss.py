import numpy as np
import torch
from torch.nn import functional as F

from lib.ops import lovasz_softmax_loss


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_one_mean_iou(image_array, label_array, num_parsing):
    hist = fast_hist(label_array, image_array, num_parsing).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / union
    return iu


class ParsingLossComputation(object):
    def __init__(self, cfg):
        self.device = torch.device(cfg.DEVICE)
        self.parsingiou_on = cfg.PARSING.PARSINGIOU_ON
        self.num_parsing = cfg.PARSING.NUM_PARSING
        self.loss_weight = cfg.PARSING.LOSS_WEIGHT
        self.lovasz_loss_weight = cfg.PARSING.LOVASZ_LOSS_WEIGHT

    def __call__(self, logits, parsing_targets):
        parsing_logits = logits[-1]
        if self.parsingiou_on:
            pred_parsings_np = parsing_logits.detach().argmax(dim=1).cpu().numpy()
            parsing_targets_np = parsing_targets.cpu().numpy()

            N = parsing_targets_np.shape[0]
            parsingiou_targets = np.zeros(N, dtype=np.float)

            for _ in range(N):
                parsing_iou = cal_one_mean_iou(parsing_targets_np[_], pred_parsings_np[_], self.num_parsing)
                parsingiou_targets[_] = np.nanmean(parsing_iou)
            parsingiou_targets = torch.from_numpy(parsingiou_targets).to(self.device, dtype=torch.float)
        else:
            parsingiou_targets = None

        parsing_targets = parsing_targets.to(self.device)
        parsing_loss = F.cross_entropy(parsing_logits, parsing_targets, reduction='mean')
        parsing_loss *= self.loss_weight

        if self.lovasz_loss_weight:
            lovasz_loss = lovasz_softmax_loss(parsing_logits, parsing_targets)
            lovasz_loss *= self.lovasz_loss_weight
            parsing_loss += lovasz_loss

        return parsing_loss, parsingiou_targets


def parsing_loss_evaluator(cfg):
    loss_evaluator = ParsingLossComputation(cfg)
    return loss_evaluator

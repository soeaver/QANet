import numpy as np

import torch
from torch.nn import functional as F

from lib.ops import lovasz_softmax_loss, l2_loss


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


class QualityLossComputation(object):
    def __init__(self, cfg):
        self.device = torch.device(cfg.DEVICE)
        self.num_parsing = cfg.PARSING.NUM_PARSING
        self.parsing_loss_weight = cfg.PARSING.QUALITY.PARSING_LOSS_WEIGHT
        self.lovasz_loss_weight = cfg.PARSING.QUALITY.LOVASZ_LOSS_WEIGHT
        self.iou_loss_weight = cfg.PARSING.QUALITY.IOU_LOSS_WEIGHT

    def __call__(self, parsing_logits, iou_pred, parsing_targets):
        b, c, h, w = parsing_logits.size()
        parsing_targets = F.interpolate(
            parsing_targets.unsqueeze(1).float(), size=(h, w), mode="nearest"
        ).type_as(parsing_targets).squeeze()

        # prepare iou targets
        pred_parsings_np = parsing_logits.detach().argmax(dim=1).cpu().numpy()
        parsing_targets_np = parsing_targets.cpu().numpy()

        N = parsing_targets_np.shape[0]
        iou_targets = np.zeros(N, dtype=np.float)

        for _ in range(N):
            parsing_iou = cal_one_mean_iou(parsing_targets_np[_], pred_parsings_np[_], self.num_parsing)
            iou_targets[_] = np.nanmean(parsing_iou)
        iou_targets = torch.from_numpy(iou_targets).to(self.device, dtype=torch.float)

        parsing_targets = parsing_targets.to(self.device)
        parsing_loss = F.cross_entropy(parsing_logits, parsing_targets)
        parsing_loss *= self.parsing_loss_weight

        if self.lovasz_loss_weight:
            lovasz_loss = lovasz_softmax_loss(parsing_logits, parsing_targets)
            lovasz_loss *= self.lovasz_loss_weight
            parsing_loss += lovasz_loss

        iou_loss = l2_loss(iou_pred[:, 0], iou_targets.detach())
        iou_loss *= self.iou_loss_weight

        return dict(loss_quality_parsing=parsing_loss, loss_quality_iou=iou_loss)


def quality_loss_evaluator(cfg):
    loss_evaluator = QualityLossComputation(cfg)
    return loss_evaluator

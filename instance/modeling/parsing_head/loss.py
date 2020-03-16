import numpy as np
import cv2

import torch
from torch.nn import functional as F

from models.ops import LovaszSoftmax
from instance.core.config import cfg


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


def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255) & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w]) & (label[:, :w - 1] != 255) & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w]) & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1]) & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge


class ParsingLossComputation(object):
    def __init__(self, device, parsingiou_on):
        self.device = torch.device(device)
        self.parsingedge_on = cfg.PARSING.PARSINGEDGE_ON
        self.parsingiou_on = parsingiou_on
        self.lavasz_loss = LovaszSoftmax()

    def __call__(self, parsing_outputs, parsing_targets):
        if self.parsingedge_on:
            parsing_logits, edge_logits = parsing_outputs
        else:
            parsing_logits = parsing_outputs[0]

        if self.parsingedge_on:
            parsing_targets_np = parsing_targets.cpu().numpy()
            N = parsing_targets_np.shape[0]
            edge_targets = np.zeros(parsing_targets_np.shape)
            for n in range(N):
                edge_targets[n] = generate_edge(parsing_targets_np[n])
            edge_targets = torch.from_numpy(edge_targets).to(self.device)

            pos_num = torch.sum(edge_targets == 1, dtype=torch.float)
            neg_num = torch.sum(edge_targets == 0, dtype=torch.float)
            weight_pos = neg_num / (pos_num + neg_num)
            weight_neg = pos_num / (pos_num + neg_num)
            edge_weights = torch.tensor([weight_neg, weight_pos]).to(self.device, dtype=torch.float)

        if self.parsingiou_on:
            pred_parsings_np = parsing_logits.detach().argmax(dim=1).cpu().numpy()
            parsing_targets_np = parsing_targets.cpu().numpy()
            N = parsing_targets_np.shape[0]
            iou_targets = np.zeros(N, dtype=np.float)
            for n in range(N):
                iou = cal_one_mean_iou(parsing_targets_np[n], pred_parsings_np[n], cfg.PARSING.NUM_PARSING)
                iou_targets[n] = np.nanmean(iou)
            iou_targets = torch.from_numpy(iou_targets).to(self.device, dtype=torch.float)

        parsing_targets = parsing_targets.to(self.device)
        parsing_loss = F.cross_entropy(parsing_logits, parsing_targets)
        lavasz_loss = self.lavasz_loss(parsing_logits, parsing_targets)
        parsing_loss *= cfg.PARSING.LOSS_WEIGHT

        if self.parsingedge_on:
            edge_loss = F.cross_entropy(edge_logits, edge_targets, edge_weights)
            return parsing_loss + lavasz_loss, edge_loss

        if not self.parsingiou_on:
            return parsing_loss, lavasz_loss
        else:
            return parsing_loss, iou_targets


def parsing_loss_evaluator():
    loss_evaluator = ParsingLossComputation(
        cfg.DEVICE, cfg.PARSING.PARSINGIOU_ON
    )
    return loss_evaluator

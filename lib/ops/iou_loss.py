import math

import torch
import torch.nn as nn


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type, box_type='lrtb'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type
        self.box_type = box_type

    def forward(self, pred, target, weight=None):
        if self.box_type == 'lrtb':
            pred_left = pred[:, 0]
            pred_top = pred[:, 1]
            pred_right = pred[:, 2]
            pred_bottom = pred[:, 3]

            target_left = target[:, 0]
            target_top = target[:, 1]
            target_right = target[:, 2]
            target_bottom = target[:, 3]

            target_area = (target_left + target_right) * (target_top + target_bottom)
            pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

            w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
            h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)

            area_intersect = w_intersect * h_intersect
            area_union = target_area + pred_area - area_intersect
            ac_uion = g_w_intersect * g_h_intersect + 1e-7

            ious = (area_intersect.clamp(min=0) + 1.0) / (area_union + 1.0)
            assert (ious > 0).all()
            gious = ious - (ac_uion - area_union) / ac_uion

            if self.loc_loss_type == 'diou' or self.loc_loss_type == 'ciou':
                target_center_x = (target_right - target_left) / 2
                target_center_y = (target_top - target_bottom) / 2
                pred_center_x = (pred_right - pred_left) / 2
                pred_center_y = (pred_top - pred_bottom) / 2

                inter_diag = (target_center_x - pred_center_x) ** 2 + (target_center_y - pred_center_y) ** 2
                outer_diag = g_w_intersect ** 2 + g_h_intersect ** 2
                u = inter_diag / outer_diag
                dious = ious - u

        elif self.box_type == 'xyxy':
            assert self.loc_loss_type != 'diou' and self.loc_loss_type != 'ciou'
            x1, y1, x2, y2 = pred.unbind(dim=-1)
            x1g, y1g, x2g, y2g = target.unbind(dim=-1)

            assert (x2 >= x1).all(), "bad box: x1 larger than x2"
            assert (y2 >= y1).all(), "bad box: y1 larger than y2"

            # Intersection keypoints
            xkis1 = torch.max(x1, x1g)
            ykis1 = torch.max(y1, y1g)
            xkis2 = torch.min(x2, x2g)
            ykis2 = torch.min(y2, y2g)

            area_intersect = torch.zeros_like(x1)
            mask = (ykis2 > ykis1) & (xkis2 > xkis1)
            area_intersect[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
            area_union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - area_intersect
            ious = area_intersect / (area_union + 1e-7)

            # smallest enclosing box
            xc1 = torch.min(x1, x1g)
            yc1 = torch.min(y1, y1g)
            xc2 = torch.max(x2, x2g)
            yc2 = torch.max(y2, y2g)

            ac_uion = (xc2 - xc1) * (yc2 - yc1)
            gious = ious - ((ac_uion - ac_uion) / (ac_uion + 1e-7))

        else:
            raise NotImplementedError

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'liou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        elif self.loc_loss_type == 'diou':
            losses = 1 - dious
        elif self.loc_loss_type == 'ciou':
            v = (4 / (math.pi ** 2)) * torch.pow((
                    torch.atan((target_left + target_right) / (target_top + target_bottom + 1e-7)) -
                    torch.atan((pred_left + pred_right) / (pred_top + pred_bottom + 1e-7))), 2)
            S = 1 - ious
            alpha = v / (S + v)
            cious = ious - (u + alpha * v)
            losses = 1 - cious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class BoundedIoULoss(nn.Module):
    def __init__(self, beta=0.2, eps=1e-3):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target, weight=None):
        pred_ctr_2x = pred[:, :2] + pred[:, 2:]
        pred_wh = pred[:, 2:] - pred[:, :2]
        with torch.no_grad():
            target_ctr_2x = target[:, :2] + target[:, 2:]
            target_wh = target[:, 2:] - target[:, :2]

        d_xy_2x = (target_ctr_2x - pred_ctr_2x).abs()

        loss_xy = torch.clamp((target_wh - d_xy_2x) / (target_wh + d_xy_2x + self.eps), min=0)
        loss_wh = torch.min(target_wh / (pred_wh + self.eps), pred_wh / (target_wh + self.eps))
        loss = 1 - torch.cat([loss_xy, loss_wh], dim=-1)

        if self.beta >= 1e-5:
            loss = torch.where(loss < self.beta, 0.5 * loss ** 2 / self.beta, loss - 0.5 * self.beta)

        if weight is not None:
            loss = loss * weight

        return loss.sum()


class MaskIOULoss(nn.Module):
    def __init__(self):
        super(MaskIOULoss, self).__init__()

    def forward(self, pred, target, weight):
        total = torch.stack([pred, target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        return loss.sum()

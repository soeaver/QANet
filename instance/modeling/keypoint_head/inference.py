import numpy as np
import torch
from torch.nn import functional as F


def get_keypoints_results(cfg, probs, targets):
    device = probs.device
    N, C, H, W = probs.shape

    ims_keypoints_max, ims_keypoints_idx = torch.max(probs.view(N, C, -1), dim=2)

    ims_x_int = ims_keypoints_idx % W
    ims_y_int = (ims_keypoints_idx - ims_x_int) // W
    ims_x_int = ims_x_int.float()
    ims_y_int = ims_y_int.float()
    ims_keypoints_max = torch.clamp(ims_keypoints_max, min=0)

    # post-processing
    if cfg.TEST.AUG.POST_PROCESS:
        for n in range(N):
            for c in range(C):
                prob = probs[n][c]
                px = int(ims_x_int[n][c])
                py = int(ims_y_int[n][c])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    ims_x_int[n][c] += torch.sign(prob[py][px + 1] - prob[py][px - 1]) * .25
                    ims_y_int[n][c] += torch.sign(prob[py + 1][px] - prob[py - 1][px]) * .25

    all_boxes = torch.cat([target.ori_bbox for target in targets]).to(device)
    xc, yc, w, h, r = all_boxes.split(1, dim=-1)
    rois = torch.cat((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), dim=-1)
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)

    width_corrections = widths / W
    height_corrections = heights / H

    x = ims_x_int * width_corrections[:, None] + offset_x[:, None]
    y = ims_y_int * height_corrections[:, None] + offset_y[:, None]

    ims_kpts = torch.stack((x, y, ims_keypoints_max), dim=2)

    kpt_index = (ims_keypoints_max >= cfg.KEYPOINT.INDEX_THRESH).to(dtype=torch.bool)
    kpt_pixel_scores = torch.sum(ims_keypoints_max * kpt_index, dim=1).to(dtype=torch.float32) \
                 / torch.clamp(torch.sum(kpt_index, dim=1).to(dtype=torch.float32), min=1e-6)

    boxes_per_image = [len(target) for target in targets]
    ims_kpts = ims_kpts.split(boxes_per_image, dim=0)
    ims_kpts = [im_kpts.cpu() for im_kpts in ims_kpts]
    kpt_pixel_scores = kpt_pixel_scores.split(boxes_per_image, dim=0)
    kpt_pixel_scores = [_.cpu() for _ in kpt_pixel_scores]

    return ims_kpts, kpt_pixel_scores

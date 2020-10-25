import numpy as np

import torch
from torch.nn import functional as F


def _do_paste_parsing(probs, boxes, im_h, im_w, skip_empty=True):
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = probs.device
    N = probs.shape[0]

    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=im_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=im_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = im_w, im_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)
    img_parsing = F.grid_sample(probs.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_parsing, (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_parsing, ()


def paste_parsing(probs, ims_probs, all_boxes, scale, ims_h_max, ims_w_max, chunks):
    device = probs.device
    xc, yc, w, h, r = all_boxes.clone().split(1, dim=-1)
    w *= scale
    h *= scale
    bbox_xxyy = torch.cat((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), dim=-1)
    for i in chunks:
        probs_chunk, spatial_inds = _do_paste_parsing(
            probs, bbox_xxyy[i], ims_h_max, ims_w_max, skip_empty=device.type == "cpu"
        )
        ims_probs[(i,) + spatial_inds] = probs_chunk

    return ims_probs


def paste_parsing_aug(cfg, probs_list, all_boxes, ims_h_max, ims_w_max, chunks):
    device = probs_list[0].device
    N, C, H, W = probs_list[0].shape
    ims_probs = [torch.zeros(N, C, ims_h_max, ims_w_max, device=device, dtype=torch.float32)
                 for _ in range(len(probs_list))]
    aug_idx = 0
    ims_probs[aug_idx] = paste_parsing(
        probs_list[aug_idx], ims_probs[aug_idx], all_boxes, 1., ims_h_max, ims_w_max, chunks
    )
    aug_idx += 1
    if cfg.TEST.AUG.H_FLIP:
        ims_probs[aug_idx] = paste_parsing(
            probs_list[aug_idx], ims_probs[aug_idx], all_boxes, 1., ims_h_max, ims_w_max, chunks
        )
        aug_idx += 1
    for scale in cfg.TEST.AUG.SCALES:
        ims_probs[aug_idx] = paste_parsing(
            probs_list[aug_idx], ims_probs[aug_idx], all_boxes, scale, ims_h_max, ims_w_max, chunks
        )
        aug_idx += 1
        if cfg.TEST.AUG.H_FLIP:
            ims_probs[aug_idx] = paste_parsing(
                probs_list[aug_idx], ims_probs[aug_idx], all_boxes, scale, ims_h_max, ims_w_max, chunks
            )
            aug_idx += 1
    ims_probs = torch.stack(ims_probs, dim=0)
    ims_probs = torch.mean(ims_probs, dim=0)
    return ims_probs


def get_parsing_results(cfg, probs, targets):
    BYTES_PER_FLOAT = 4
    # TODO: This memory limit may be too much or too little. It would be better to
    # determine it based on available resources.
    GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

    device = probs[0].device
    ims_w_max = max([target.size[0] for target in targets])
    ims_h_max = max([target.size[1] for target in targets])
    all_boxes = torch.cat([target.ori_bbox for target in targets]).to(device)
    N, C, H, W = probs[0].shape
    value_thresh = 1.0 / C

    num_chunks = N if device.type == "cpu" else \
        int(np.ceil(N * int(ims_h_max * ims_w_max) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
    assert num_chunks <= N, "Default GPU_MEM_LIMIT in is too small; try increasing it"

    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
    ims_probs = paste_parsing_aug(cfg, probs, all_boxes, ims_h_max, ims_w_max, chunks)

    inst_max, inst_idx = torch.max(ims_probs, dim=1)
    ims_parsings = inst_idx.to(dtype=torch.uint8) * (inst_max >= value_thresh).to(dtype=torch.bool)

    # high confidence mask (hcm)
    inst_hcm = (inst_max >= cfg.PARSING.PIXEL_SCORE_TH).to(dtype=torch.bool)
    # high confidence value (hcv)
    inst_hcv = torch.sum(inst_max * inst_hcm, dim=[1, 2]).to(dtype=torch.float32)
    inst_hcm_num = torch.clamp(torch.sum(inst_hcm, dim=[1, 2]).to(dtype=torch.float32), min=1e-6)

    instance_pixel_scores = inst_hcv / inst_hcm_num

    category_pixel_scores = torch.ones((N, C - 1), device=device, dtype=torch.float32)
    for c in range(1, cfg.PARSING.NUM_PARSING):
        cate_max = inst_max * (inst_idx == c).to(dtype=torch.bool)

        cate_hcm = (cate_max >= cfg.PARSING.PIXEL_SCORE_TH).to(dtype=torch.bool)
        cate_hcv = torch.sum(cate_max * cate_hcm, dim=[1, 2]).to(dtype=torch.float32)
        cate_hcm_num = torch.clamp(torch.sum(cate_hcm, dim=[1, 2]).to(dtype=torch.float32), min=1e-6)

        category_pixel_scores[:, c - 1] = cate_hcv / cate_hcm_num

    boxes_per_image = [len(target) for target in targets]
    ims_parsings = ims_parsings.split(boxes_per_image, dim=0)
    ims_parsings = [im_parsings.cpu() for im_parsings in ims_parsings]
    instance_pixel_scores = instance_pixel_scores.split(boxes_per_image, dim=0)
    instance_pixel_scores = [_.cpu() for _ in instance_pixel_scores]
    category_pixel_scores = category_pixel_scores.split(boxes_per_image, dim=0)
    category_pixel_scores = [_.cpu() for _ in category_pixel_scores]

    return ims_parsings, instance_pixel_scores, category_pixel_scores

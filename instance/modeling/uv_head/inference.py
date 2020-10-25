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


def get_uv_results(cfg, probs, targets):
    AnnIndex, Index_UV, U_uv, V_uv = probs

    device = AnnIndex.device
    ims_w_max = max([target.size[0] for target in targets])
    ims_h_max = max([target.size[1] for target in targets])
    all_boxes = torch.cat([target.ori_bbox for target in targets]).to(device)
    xc, yc, w, h, r = all_boxes.split(1, dim=-1)
    all_boxes = torch.cat((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), dim=-1)

    N, C, H, W = AnnIndex.shape
    _, K, _, _ = Index_UV.shape

    num_chunks = N if device.type == "cpu" else 1
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
    ims_AnnIndex = torch.zeros(N, C, ims_h_max, ims_w_max, device=device, dtype=torch.float32)
    ims_Index_UV = torch.zeros(N, K, ims_h_max, ims_w_max, device=device, dtype=torch.float32)
    ims_U_uv = torch.zeros(N, K, ims_h_max, ims_w_max, device=device, dtype=torch.float32)
    ims_V_uv = torch.zeros(N, K, ims_h_max, ims_w_max, device=device, dtype=torch.float32)

    for i in chunks:
        AnnIndex_chunk, spatial_inds = _do_paste_parsing(
            AnnIndex[i], all_boxes[i], ims_h_max, ims_w_max, skip_empty=device.type == "cpu"
        )
        ims_AnnIndex[(i,) + spatial_inds] = AnnIndex_chunk

        Index_UV_chunk, spatial_inds = _do_paste_parsing(
            Index_UV[i], all_boxes[i], ims_h_max, ims_w_max, skip_empty=device.type == "cpu"
        )
        ims_Index_UV[(i,) + spatial_inds] = Index_UV_chunk

        U_uv_chunk, spatial_inds = _do_paste_parsing(
            U_uv[i], all_boxes[i], ims_h_max, ims_w_max, skip_empty=device.type == "cpu"
        )
        ims_U_uv[(i,) + spatial_inds] = U_uv_chunk

        V_uv_chunk, spatial_inds = _do_paste_parsing(
            V_uv[i], all_boxes[i], ims_h_max, ims_w_max, skip_empty=device.type == "cpu"
        )
        ims_V_uv[(i,) + spatial_inds] = V_uv_chunk

    ims_Index_max, ims_Index_UV = torch.max(ims_Index_UV, dim=1)
    _, ims_AnnIndex = torch.max(ims_AnnIndex, dim=1)
    ims_Index_UV = ims_Index_UV * (ims_AnnIndex > 0).float()
    ims_Index_max = ims_Index_max * (ims_AnnIndex > 0).float()

    uv_index = (ims_Index_max >= cfg.UV.INDEX_THRESH).to(dtype=torch.bool)
    uv_pixel_scores = torch.sum(ims_Index_max * uv_index, dim=[1, 2]).to(dtype=torch.float32) \
                      / torch.clamp(torch.sum(uv_index, dim=[1, 2]).to(dtype=torch.float32), min=1e-6)

    boxes_per_image = [len(target) for target in targets]
    ims_Index_UV = ims_Index_UV.split(boxes_per_image, dim=0)
    ims_Index_UV = [im_Index_UV.cpu() for im_Index_UV in ims_Index_UV]
    ims_U_uv = ims_U_uv.split(boxes_per_image, dim=0)
    ims_U_uv = [_.cpu() for _ in ims_U_uv]
    ims_V_uv = ims_V_uv.split(boxes_per_image, dim=0)
    ims_V_uv = [_.cpu() for _ in ims_V_uv]
    uv_pixel_scores = uv_pixel_scores.split(boxes_per_image, dim=0)
    uv_pixel_scores = [_.cpu() for _ in uv_pixel_scores]

    return ims_Index_UV, ims_U_uv, ims_V_uv, uv_pixel_scores

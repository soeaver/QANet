import numpy as np

import torch

from lib.data.structures.bounding_box import BoxList
from lib.ops import box_ml_voting as _box_ml_voting
from lib.ops import box_voting as _box_voting
from lib.ops import ml_nms as _box_ml_nms
from lib.ops import ml_soft_nms as _box_ml_soft_nms
from lib.ops import nms as _box_nms
from lib.ops import soft_nms as _box_soft_nms

TO_REMOVE = 0


def boxlist_nms(boxlist, nms_thresh, topk=0, score_field="scores", idxs=None):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        topk (int): if > 0, then only the top k are kept
            after non-maximum suppression
        score_field (str)
        idxs (Tensor)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    if idxs is not None:
        # batched nms
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes = boxes + offsets[:, None]
    keep = _box_nms(boxes, score, nms_thresh)
    if keep.size(0) > topk > 0:
        keep = keep[: topk]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def boxlist_ml_nms(boxlist, nms_thresh, topk=0, score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        topk (int): if > 0, then only the top k are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    labels = boxlist.get_field(label_field)
    keep = _box_ml_nms(boxes, scores, labels, nms_thresh, topk)
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def boxlist_soft_nms(boxlist, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear', score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if overlap_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox.cpu()
    score = boxlist.get_field(score_field).cpu()
    dets, scores, _ = _box_soft_nms(boxes, score, sigma, overlap_thresh, score_thresh, method)
    boxlist = BoxList(dets.cuda(), boxlist.size, mode="xyxy")
    boxlist.add_field("scores", scores.cuda())
    return boxlist.convert(mode)


def boxlist_ml_soft_nms(boxlist, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear', topk=0, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if overlap_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox.cpu()
    score = boxlist.get_field(score_field).cpu()
    label = boxlist.get_field("labels").cpu()
    dets, scores, labels, _ = _box_ml_soft_nms(boxes, score, label, sigma, overlap_thresh, score_thresh, method, topk)
    boxlist = BoxList(dets.cuda(), boxlist.size, mode="xyxy")
    boxlist.add_field("scores", scores.cuda())
    boxlist.add_field("labels", labels.cuda())
    return boxlist.convert(mode)


def boxlist_box_voting(top_boxlist, all_boxlist, thresh, scoring_method='ID', beta=1.0, score_field="scores"):
    if thresh <= 0:
        return top_boxlist
    mode = top_boxlist.mode
    top_boxes = top_boxlist.convert("xyxy").bbox
    all_boxes = all_boxlist.convert("xyxy").bbox
    top_score = top_boxlist.get_field(score_field)
    all_score = all_boxlist.get_field(score_field)
    boxes, scores = _box_voting(top_boxes, top_score, all_boxes, all_score, thresh, scoring_method, beta)
    boxlist = BoxList(boxes, all_boxlist.size, mode="xyxy")
    boxlist.add_field("scores", scores)
    return boxlist.convert(mode)


def boxlist_box_ml_voting(top_boxlist, all_boxlist, thresh, scoring_method='ID', beta=1.0, score_field="scores"):
    if thresh <= 0:
        return top_boxlist
    mode = top_boxlist.mode
    top_boxes = top_boxlist.convert("xyxy").bbox
    all_boxes = all_boxlist.convert("xyxy").bbox
    top_score = top_boxlist.get_field(score_field)
    all_score = all_boxlist.get_field(score_field)
    top_label = top_boxlist.get_field("labels")
    all_label = all_boxlist.get_field("labels")
    boxes, scores, labels = _box_ml_voting(top_boxes, top_score, top_label, all_boxes, all_score, all_label, thresh, scoring_method, beta)
    boxlist = BoxList(boxes, all_boxlist.size, mode="xyxy")
    boxlist.add_field("scores", scores)
    boxlist.add_field("labels", labels)
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero(as_tuple=False).squeeze(1)
    return boxlist[keep]


def remove_boxes_by_center(boxlist, crop_region):
    xyxy_boxes = boxlist.convert("xyxy").bbox
    left, up, right, bottom = crop_region
    # center of boxes should inside the crop img
    centers = (xyxy_boxes[:, :2] + xyxy_boxes[:, 2:]) / 2
    keep = ((centers[:, 0] > left) & (centers[:, 1] > up) & (centers[:, 0] < right) & (centers[:, 1] < bottom)
            ).nonzero(as_tuple=False).squeeze(1)
    return boxlist[keep]


def remove_boxes_by_overlap(ori_targets, crop_targets, iou_th):
    ori_targets.size = crop_targets.size
    iou_matrix = boxlist_iou(ori_targets, crop_targets)
    iou_list = torch.diag(iou_matrix, diagonal=0)
    keep = (iou_list >= iou_th).nonzero(as_tuple=False).squeeze(1)
    return crop_targets[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2, mode="iou"):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      boxlist1: (BoxList) bounding boxes, sized [N,4].
      boxlist2: (BoxList) bounding boxes, sized [M,4].
      mode: 'iou' or 'iof'

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    assert mode in ['iou', 'iof']
    if boxlist1.size != boxlist2.size:
        raise RuntimeError("boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    if mode == 'iou':
        iou = inter / (area1[:, None] + area2 - inter)
    else:
        iou = inter / area1[:, None]
    return iou


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_partly_overlap(boxlist1, boxlist2):
    if boxlist1.size != boxlist2.size:
        raise RuntimeError("boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    overlap = iou > 0
    not_complete_overlap = (inter - area1[:, None]) * (inter - area2[None, :]) != 0
    partly_overlap = overlap * not_complete_overlap

    return partly_overlap


def boxlist_overlap(boxlist1, boxlist2):
    if boxlist1.size != boxlist2.size:
        raise RuntimeError("boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    overlap = iou > 0

    return overlap


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def boxes_to_masks(boxes, h, w, padding=0.0):
    n = boxes.shape[0]
    boxes = boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    b_w = x2 - x1
    b_h = y2 - y1
    x1 = torch.clamp(x1 - 1 - b_w * padding, min=0)
    x2 = torch.clamp(x2 + 1 + b_w * padding, max=w)
    y1 = torch.clamp(y1 - 1 - b_h * padding, min=0)
    y2 = torch.clamp(y2 + 1 + b_h * padding, max=h)

    rows = torch.arange(w, device=boxes.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
    cols = torch.arange(h, device=boxes.device, dtype=x1.dtype).view(1, -1, 1).expand(n, h, w)

    masks_left = rows >= x1.view(-1, 1, 1)
    masks_right = rows < x2.view(-1, 1, 1)
    masks_up = cols >= y1.view(-1, 1, 1)
    masks_down = cols < y2.view(-1, 1, 1)

    masks = masks_left * masks_right * masks_up * masks_down

    return masks


def crop_by_box(masks, box, padding=0.0):
    n, h, w = masks.size()

    b_w = box[:, 2] - box[:, 0]
    b_h = box[:, 3] - box[:, 1]
    x1 = torch.clamp(box[:, 0] - b_w * padding - 1, min=0)
    x2 = torch.clamp(box[:, 2] + b_w * padding + 1, max=w - 1)
    y1 = torch.clamp(box[:, 1] - b_h * padding - 1, min=0)
    y2 = torch.clamp(box[:, 3] + b_h * padding + 1, max=h - 1)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(n, h, w)

    masks_left = rows >= x1.view(n, 1, 1)
    masks_right = rows < x2.view(n, 1, 1)
    masks_up = cols >= y1.view(n, 1, 1)
    masks_down = cols < y2.view(n, 1, 1)

    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float(), crop_mask


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update

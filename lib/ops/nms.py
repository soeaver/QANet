import torch
from torch.cuda.amp import custom_fwd
from torchvision.ops import nms as thv_nms

from lib.ops import _C

SOFT_NMS_METHODS = {'hard': 0, 'linear': 1, 'gaussian': 2}

# Only valid with fp32 inputs - give AMP the hint
nms = custom_fwd(thv_nms, cast_inputs=torch.float32)
ml_nms = custom_fwd(_C.ml_nms, cast_inputs=torch.float32)
nms_rotated = custom_fwd(_C.nms_rotated, cast_inputs=torch.float32)
poly_nms = custom_fwd(_C.poly_nms, cast_inputs=torch.float32)


@custom_fwd(cast_inputs=torch.float32)
def soft_nms(dets, scores, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'):
    """ Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503. """

    assert method in SOFT_NMS_METHODS, 'Unknown soft_nms method: {}'.format(method)

    return _C.soft_nms(
        dets,
        scores,
        sigma,
        overlap_thresh,
        score_thresh,
        SOFT_NMS_METHODS[method]
    )


@custom_fwd(cast_inputs=torch.float32)
def ml_soft_nms(dets, scores, labels, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear', topk=0):    
    assert method in SOFT_NMS_METHODS, 'Unknown soft_nms method: {}'.format(method)

    return _C.ml_soft_nms(
        dets,
        scores,
        labels,
        sigma,
        overlap_thresh,
        score_thresh,
        SOFT_NMS_METHODS[method],
        topk
    )

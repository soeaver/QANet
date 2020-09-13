from apex import amp
from torchvision.ops import nms as thv_nms

from lib.ops import _C

SOFT_NMS_METHODS = {'hard': 0, 'linear': 1, 'gaussian': 2}


# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(thv_nms)
ml_nms = amp.float_function(_C.ml_nms)
nms_rotated = amp.float_function(_C.nms_rotated)
poly_nms = amp.float_function(_C.poly_nms)


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

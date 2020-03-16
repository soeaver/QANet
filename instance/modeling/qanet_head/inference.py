import cv2
import numpy as np

from instance.utils.transforms import get_affine_transform, affine_transform
from instance.core.config import cfg


def qanet_results(parsing, image_info, center, scale):
    results = []
    scores = []
    for img_idx in range(parsing.shape[0]):
        heatmap_h = parsing[img_idx].shape[1]
        heatmap_w = parsing[img_idx].shape[2]

        trans = get_affine_transform(center[img_idx], scale[img_idx], 0, [heatmap_w, heatmap_h], inv=1)

        _parsing = parsing[img_idx].transpose((1, 2, 0))
        im_h, im_w = image_info[img_idx]

        _parsing = cv2.warpAffine(
            _parsing,
            trans,
            (im_w, im_h),
            flags=cv2.INTER_LINEAR
        )

        _parsing_max = np.max(_parsing, axis=2)
        _parsing = np.argmax(_parsing, axis=2).astype(np.uint8)

        _index = np.where(_parsing_max >= cfg.QANET.INDEX_THRESH, 1, 0)
        score = np.sum(_parsing_max * _index) / (np.sum(_index) + 1e-6)

        results.append(_parsing)
        scores.append(score)

    scores = np.array(scores)
    return results, scores

import cv2
import numpy as np

import pycocotools.mask as mask_util

from pet.instance.utils.transforms import get_affine_transform, affine_transform
from pet.instance.core.config import cfg


def masks_results(masks, image_info, center, scale, labels):
    threshold = 0.5
    results = []
    for img_idx in range(masks.shape[0]):
        heatmap_h = masks[img_idx].shape[1]
        heatmap_w = masks[img_idx].shape[2]

        trans = get_affine_transform(center[img_idx], scale[img_idx], 0, [heatmap_w, heatmap_h], inv=1)

        mask = masks[img_idx, labels[img_idx]]
        im_h, im_w = image_info[img_idx]

        mask = cv2.warpAffine(
            mask,
            trans,
            (im_w, im_h),
            flags=cv2.INTER_LINEAR)

        mask = np.array(mask > threshold, dtype=np.uint8)

        # Get RLE encoding used by the COCO evaluation API
        rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        # For dumping to json, need to decode the byte string.
        # https://github.com/cocodataset/cocoapi/issues/70
        rle['counts'] = rle['counts'].decode('ascii')

        results.append(rle)

    return results

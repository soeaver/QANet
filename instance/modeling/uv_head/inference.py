import cv2
import numpy as np

import pycocotools.mask as mask_util

from instance.utils.transforms import get_affine_transform, affine_transform
from instance.core.config import cfg


def uvs_results(outputs, image_info, boxes, center, scale):
    AnnIndex, Index_UV, U_uv, V_uv = outputs

    results = []
    scores = []
    heatmap_h = AnnIndex.shape[2]
    heatmap_w = AnnIndex.shape[3]

    for img_idx in range(AnnIndex.shape[0]):
        trans = get_affine_transform(center[img_idx], scale[img_idx], 0, [heatmap_w, heatmap_h], inv=1)

        im_h, im_w = image_info[img_idx]

        # preds[ind] axes are CHW; bring p axes to HWC
        CurAnnIndex = AnnIndex[img_idx].transpose(1, 2, 0)
        CurIndex_UV = Index_UV[img_idx].transpose(1, 2, 0)
        CurU_uv = U_uv[img_idx].transpose(1, 2, 0)
        CurV_uv = V_uv[img_idx].transpose(1, 2, 0)

        CurAnnIndex = cv2.warpAffine(
            CurAnnIndex,
            trans,
            (im_w, im_h),
            flags=cv2.INTER_LINEAR)

        CurIndex_UV = cv2.warpAffine(
            CurIndex_UV,
            trans,
            (im_w, im_h),
            flags=cv2.INTER_LINEAR)

        CurU_uv = cv2.warpAffine(
            CurU_uv,
            trans,
            (im_w, im_h),
            flags=cv2.INTER_LINEAR)

        CurV_uv = cv2.warpAffine(
            CurV_uv,
            trans,
            (im_w, im_h),
            flags=cv2.INTER_LINEAR)

        # Bring Cur_Preds axes back to CHW
        CurAnnIndex = CurAnnIndex.transpose(2, 0, 1)
        CurIndex_UV = CurIndex_UV.transpose(2, 0, 1)
        CurU_uv = CurU_uv.transpose(2, 0, 1)
        CurV_uv = CurV_uv.transpose(2, 0, 1)

        # Removed squeeze calls due to singleton dimension issues
        CurIndex_max = np.max(CurIndex_UV, axis=0)
        CurAnnIndex = np.argmax(CurAnnIndex, axis=0)
        CurIndex_UV = np.argmax(CurIndex_UV, axis=0)
        CurIndex_UV = CurIndex_UV * (CurAnnIndex > 0).astype(np.float32)
        CurIndex_max = CurIndex_max * (CurAnnIndex > 0).astype(np.float32)

        _index = np.where(CurIndex_max > cfg.UV.INDEX_THRESH, 1, 0)
        score = np.sum(CurIndex_max * _index) / np.sum(_index)
        scores.append(score)

        x1 = int(boxes[img_idx][0])
        y1 = int(boxes[img_idx][1])
        x2 = int(x1 + np.maximum(0., boxes[img_idx][2]))
        y2 = int(y1 + np.maximum(0., boxes[img_idx][3]))

        output = np.zeros([3, int(y2 - y1), int(x2 - x1)], dtype=np.float32)
        output[0] = CurIndex_UV[y1:y2, x1:x2]

        outputU = np.zeros([im_h, im_w], dtype=np.float32)
        outputV = np.zeros([im_h, im_w], dtype=np.float32)

        for part_id in range(1, 25):
            CurrentU = CurU_uv[part_id]
            CurrentV = CurV_uv[part_id]
            outputU[CurIndex_UV == part_id] = CurrentU[CurIndex_UV == part_id]
            outputV[CurIndex_UV == part_id] = CurrentV[CurIndex_UV == part_id]
        output[1] = outputU[y1:y2, x1:x2]
        output[2] = outputV[y1:y2, x1:x2]
        results.append(output.copy())
    scores = np.array(scores)
    return results, scores

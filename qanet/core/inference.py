import numpy as np
import pycocotools.mask as mask_util
import torch

from qanet.core.config import get_cfg, infer_cfg
from qanet.core.test import TestEngine
from lib.data.structures.instance_box import InstanceBox


class Inference(object):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.test_engine = TestEngine(cfg, model)

    def __call__(self, img, boxes):
        if len(boxes) == 0:
            return None
        img = np.ascontiguousarray(img)
        boxes = torch.as_tensor(boxes)[:, :4]
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]

        classes = torch.zeros(len(boxes)).int()
        target = InstanceBox(boxes, classes, (img.shape[1], img.shape[0]))
        test_size = self.cfg.TEST.SCALE
        aspect_ratio = test_size[0] * 1.0 / test_size[1]
        target.convert(aspect_ratio, 1.0)
        results = self.test_engine([img], [target])

        ims_dets = np.hstack(
            (target.im_bbox.numpy(), target.scores.numpy()[:, np.newaxis])
        ).astype(np.float32, copy=False)

        ims_labels = target.labels.tolist()

        if self.cfg.MODEL.MASK_ON and 'mask' in results.keys():
            bitmasks = results['mask']['ims_bitmasks'][0]
            im_masks = []
            for j in range(len(bitmasks)):
                # Too slow.
                # Get RLE encoding used by the COCO evaluation API
                rle = mask_util.encode(np.array(bitmasks[j][:, :, np.newaxis], dtype=np.uint8, order='F'))[0]
                # For dumping to json, need to decode the byte string.
                # https://github.com/cocodataset/cocoapi/issues/70
                rle['counts'] = rle['counts'].decode('ascii')
                im_masks.append(rle)
        else:
            im_masks = None

        if self.cfg.MODEL.KEYPOINT_ON and 'keypoints' in results.keys():
            keypoints = results['keypoints']['ims_kpts'][0].numpy()
            ims_kpts = keypoints.transpose((0, 2, 1))
        else:
            ims_kpts = None

        if self.cfg.MODEL.PARSING_ON and 'parsing' in results.keys():
            parsings = results['parsing']['ims_parsings'][0]
            ims_parss = parsings.numpy()
        else:
            ims_parss = None

        if self.cfg.MODEL.UV_ON and 'uv' in results.keys():
            ims_Index_UV = results['uv']['ims_Index_UV']
            ims_U_uv = results['uv']['ims_U_uv']
            ims_V_uv = results['uv']['ims_V_uv']
            h, w = ims_Index_UV[0].shape[1:]

            uvs = []
            Index_UV = ims_Index_UV[0].numpy()
            U_uv = ims_U_uv[0].numpy()
            V_uv = ims_V_uv[0].numpy()

            for ind, entry in enumerate(target.im_bbox.numpy()):
                x1 = int(entry[0])
                y1 = int(entry[1])
                x2 = int(entry[2])
                y2 = int(entry[3])

                output = np.zeros([3, int(y2 - y1), int(x2 - x1)], dtype=np.float32)
                output[0] = Index_UV[ind][y1:y2, x1:x2]

                outputU = np.zeros([h, w], dtype=np.float32)
                outputV = np.zeros([h, w], dtype=np.float32)

                for part_id in range(1, self.cfg.UV.NUM_PATCHES + 1):
                    CurrentU = U_uv[ind][part_id]
                    CurrentV = V_uv[ind][part_id]
                    outputU[Index_UV[ind] == part_id] = CurrentU[Index_UV[ind] == part_id]
                    outputV[Index_UV[ind] == part_id] = CurrentV[Index_UV[ind] == part_id]
                output[1] = outputU[y1:y2, x1:x2]
                output[2] = outputV[y1:y2, x1:x2]
                uvs.append(output.copy())
            ims_uvs = uvs
        else:
            ims_uvs = None

        return ims_dets, ims_labels, im_masks, ims_kpts, ims_parss, ims_uvs

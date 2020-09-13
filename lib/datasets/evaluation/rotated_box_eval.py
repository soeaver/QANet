import numpy as np
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lib.ops.rotatedboxes_ops import rotatedboxes_iou


class RotatedCOCOeval(COCOeval):
    def compute_iou_dt_gt(self, dt, gt, is_crowd):
        assert all(c == 0 for c in is_crowd)
        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]
        device = g.device if isinstance(g, torch.Tensor) else torch.device("cpu")
        g = torch.as_tensor(g, dtype=torch.float32, device=device).reshape(-1, 5)
        d = torch.as_tensor(d, dtype=torch.float32, device=device).reshape(-1, 5)
        return rotatedboxes_iou(d, g)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0: p.maxDets[-1]]

        assert p.iouType == "bbox", "unsupported iouType for iou computation"
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = self.compute_iou_dt_gt(dt, gt, iscrowd)
        return ious

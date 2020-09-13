import numpy as np

import torch

from lib.datasets.structures.bounding_box import BoxList
from lib.ops.boxlist_ops import boxlist_iou


class BoxProposalEvaluator:
    """
    Evaluate box proposal
    """

    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox'):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        if not iouType:
            print('iouType not specified. use default iouType bbox')
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.params = {}  # evaluation parameters
        self.params = Params(iouType=iouType)  # parameters
        self.stats = []  # result summarization
        self.thresholds = torch.from_numpy(self.params.iouThrs)
        self.image_ids = []
        for prediction in self.cocoDt:
            # all gather will disorder image ids
            self.image_ids.append(prediction.get_field("image_id"))
            del prediction.extra_fields["image_id"]

    def compute(self, limit, area_range):
        gt_overlaps = []
        num_pos = 0
        for image_id, prediction in zip(self.image_ids, self.cocoDt):
            original_id = self.cocoGt.id_to_img_map[image_id]
            img_info = self.cocoGt.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = prediction.resize((image_width, image_height))

            # sort predictions in descending order
            inds = prediction.get_field("objectness").sort(descending=True)[1]
            prediction = prediction[inds]

            ann_ids = self.cocoGt.coco.getAnnIds(imgIds=original_id)
            anno = self.cocoGt.coco.loadAnns(ann_ids)
            gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
            gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert("xyxy")
            gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

            if len(gt_boxes) == 0:
                continue

            valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
            gt_boxes = gt_boxes[valid_gt_inds]

            num_pos += len(gt_boxes)

            if len(gt_boxes) == 0:
                continue

            if len(prediction) == 0:
                continue

            if limit is not None and len(prediction) > limit:
                prediction = prediction[:limit]

            overlaps = boxlist_iou(prediction, gt_boxes)
            _gt_overlaps = torch.zeros(len(gt_boxes))
            for j in range(min(len(prediction), len(gt_boxes))):
                # find which proposal box maximally covers each gt box
                # and get the iou amount of coverage for each gt box
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)

                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ovr, gt_ind = max_overlaps.max(dim=0)
                assert gt_ovr >= 0
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            # append recorded iou coverage level
            gt_overlaps.append(_gt_overlaps)

        gt_overlaps = torch.cat(gt_overlaps, dim=0)
        gt_overlaps, _ = torch.sort(gt_overlaps)

        if self.thresholds is None:
            step = 0.05
            self.thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
        recalls = torch.zeros_like(self.thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(self.thresholds):
            recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {"ar": ar, "recalls": recalls, "thresholds": self.thresholds,
                "gt_overlaps": gt_overlaps, "num_pos": num_pos}

    def evaluate(self):
        print('Evaluating bbox proposals')
        for maxDet in self.params.maxDets:
            for idx, area in enumerate(self.params.areaRng):
                stat = self.compute(maxDet, area)
                stat.update(dict(maxDet=maxDet, area=self.params.areaRngLbl[idx]))
                self.stats.append(stat)

    def accumulate(self, p=None):
        pass

    def summarize(self):
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Recall'
        typeStr = '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(self.params.iouThrs[0], self.params.iouThrs[-1])

        for stat in self.stats:
            print(iStr.format(titleStr, typeStr, iouStr, stat['area'], stat['maxDet'], stat['ar']))

    def __str__(self):
        self.summarize()


class Params:
    """
    Params for coco evaluation api
    """

    def setProposalParams(self):
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True, dtype=np.float32)
        self.maxDets = [100, 1000]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='bbox'):
        if iouType == 'bbox':
            self.setProposalParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType

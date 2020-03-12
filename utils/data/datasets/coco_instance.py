import os
import cv2
import json
import numpy as np
from pycocotools.coco import COCO

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data as data

from utils.data.structures.instance import Instance
from utils.misc import logging_rank
from utils.comm import get_rank


def has_valid_annotation(obj, entry):
    # if not sperate instance
    if obj["iscrowd"] != 0:
        return False
    # if bbox is not clean
    width = entry['width']
    height = entry['height']
    x, y, w, h = obj['bbox']
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if obj['area'] <= 0 or x2 < x1 or y2 < y1:
        return False
    return True


def has_valid_person(obj):
    # if category is not person
    if obj['category_id'] != 1:
        return False
    return True


def has_valid_keypoint(obj):
    # if keypoints not exist
    if max(obj['keypoints']) == 0:
        return False
    return True


def has_valid_densepose(obj):
    if 'dp_masks' not in obj.keys():
        return False
    return True


class COCOInstanceDataset(data.Dataset):
    def __init__(self, ann_file, root, bbox_file, image_thresh, ann_types, transforms=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ann_types = ann_types
        self.transforms = transforms
        self.ids = sorted(list(self.coco.imgs.keys()))

        self.bbox_file = bbox_file
        self.use_gt_bbox = False if bbox_file else True
        self.image_thresh = image_thresh

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        if self.use_gt_bbox:
            ids, anno_idx = self._load_annotations()
        else:
            ids, anno_idx = self._load_detection_results()

        self.ids = ids
        self.anno_idx = anno_idx
        category_ids = self.coco.getCatIds()
        categories = [c['name'] for c in self.coco.loadCats(category_ids)]
        self.classes = ['__background__'] + categories

    def __getitem__(self, idx):
        if self.use_gt_bbox:
            img_path, box, instance, score = self._get_item_annotations(idx)
        else:
            img_path, box, instance, score = self._get_item_detection_results(idx)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target = Instance(box, img.shape[:2], self.ids[idx], score, self.ann_types, instance)

        # transform
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def _load_annotations(self):
        ids = []
        anno_idx = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            entry = self.coco.loadImgs(img_id)[0]
            for obj in anno:
                if has_valid_annotation(obj, entry):
                    if 'keypoints' in self.ann_types or 'parsing' in self.ann_types or 'uv' in self.ann_types:
                        if not has_valid_person(obj):
                            continue

                        if 'keypoints' in self.ann_types:
                            if not has_valid_keypoint(obj):
                                continue

                        if 'uv' in self.ann_types:
                            if not has_valid_densepose(obj):
                                continue

                    ids.append(img_id)
                    anno_idx.append(obj['id'])

        logging_rank('Load {} samples'.format(len(ids)), local_rank=get_rank())

        return ids, anno_idx

    def _load_detection_results(self):
        ids = []
        anno_idx = []
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        self.all_boxes = all_boxes
        logging_rank('=> Total boxes: {}'.format(len(all_boxes)), local_rank=get_rank())
        num_boxes = 0
        for index, entry in enumerate(all_boxes):
            # if entry['category_id'] != 1:
            #     continue
            if entry['score'] < self.image_thresh:
                continue
            num_boxes += 1
            ids.append(entry['image_id'])
            anno_idx.append(index)
        logging_rank('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thresh, num_boxes), local_rank=get_rank()
        )

        return ids, anno_idx

    def _get_item_annotations(self, idx):
        anno = self.coco.loadAnns(self.anno_idx[idx])[0]
        path = self.coco.loadImgs(self.ids[idx])[0]['file_name']
        img_path = os.path.join(self.root, path)
        box = anno['bbox']
        instance = {}
        if 'mask' in self.ann_types:
            _class = self.json_category_id_to_contiguous_id[anno["category_id"]]
            instance['mask'] = [anno['segmentation'], _class]
        if 'keypoints' in self.ann_types:
            instance['keypoints'] = anno['keypoints']
        if 'parsing' in self.ann_types:
            parsing_root = self.root.replace('img', 'parsing')
            instance['parsing'] = os.path.join(parsing_root, anno['parsing'])
        if 'uv' in self.ann_types:
            instance['uv'] = [anno['dp_x'], anno['dp_y'], anno['dp_I'], anno['dp_U'], anno['dp_V'], anno['dp_masks']]
        score = 1

        return img_path, box, instance, score

    def _get_item_detection_results(self, idx):
        path = self.coco.loadImgs(self.ids[idx])[0]['file_name']
        img_path = os.path.join(self.root, path)
        box = self.all_boxes[self.anno_idx[idx]]['bbox']
        instance = {}
        if 'mask' in self.ann_types:
            json_class = self.all_boxes[self.anno_idx[idx]]['category_id']
            _class = self.json_category_id_to_contiguous_id[json_class]
            instance['mask'] = ['mask_temp', _class]
        if 'keypoints' in self.ann_types:
            instance['keypoints'] = 'keypoints_temp'
        if 'parsing' in self.ann_types:
            instance['parsing'] = 'parsing_temp'
        if 'uv' in self.ann_types:
            instance['uv'] = 'uv_temp'
        score = self.all_boxes[self.anno_idx[idx]]['score']

        return img_path, box, instance, score

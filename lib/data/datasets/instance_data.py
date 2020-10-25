import os
import numpy as np
from PIL import Image, ImageFile
from pycocotools.coco import COCO

import torch
import torch.utils.data as data

from lib.data.structures.instance import Instance, Parsing
from lib.data.structures.instance_box import InstanceBox
from lib.utils.misc import logging_rank

ImageFile.LOAD_TRUNCATED_IMAGES = True


def has_valid_annotation(obj, entry):
    # if not sperate instance
    if 'iscrowd' in obj:
        if obj["iscrowd"] != 0:
            return False
    # if bbox is not clean
    width = entry['width']
    height = entry['height']
    x, y, w, h = obj['bbox']
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width, x1 + np.max((0, w))))
    y2 = np.min((height, y1 + np.max((0, h))))
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
    def __init__(self, ann_file, root, bbox_file, image_thresh, ann_types, transforms=None, extra_fields={}):
        self.root = root
        self.coco = COCO(ann_file)
        self.bbox_file = bbox_file
        self.image_thresh = image_thresh
        self.ann_types = ann_types
        self.transforms = transforms
        self.extra_fields = extra_fields

        ids = sorted(self.coco.imgs.keys())
        self.ids = []
        self.ann_ids = []
        for img_id in ids:
            ann_ids_per_image = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            ann = self.coco.loadAnns(ann_ids_per_image)
            entry = self.coco.loadImgs(img_id)[0]
            for obj in ann:
                if has_valid_annotation(obj, entry):
                    if 'keypoints' in self.ann_types or 'parsing' in self.ann_types or 'uv' in self.ann_types:
                        if not has_valid_person(obj):
                            continue
                    if 'keypoints' in self.ann_types and not has_valid_keypoint(obj):
                        continue
                    if 'uv' in self.ann_types and not has_valid_densepose(obj):
                        continue
                    self.ids.append(img_id)
                    self.ann_ids.append(obj['id'])
        logging_rank('Load {} samples'.format(len(ids)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        category_ids = sorted(self.coco.getCatIds())
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(category_ids)}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        category_ids = [c['name'] for c in self.coco.loadCats(category_ids)]
        self.classes = category_ids
        if 'parsing' in self.ann_types:
            Parsing.FLIP_MAP = self.extra_fields['flip_map'] if 'flip_map' in self.extra_fields else ()

    def __getitem__(self, idx):
        ann = self.coco.loadAnns(self.ann_ids[idx])[0]
        try:
            file_name = self.coco.loadImgs(self.ids[idx])[0]['file_name']
        except:
            file_name = self.coco.loadImgs(self.ids[idx])[0]['coco_url'].split('.org/')[-1]
        img_path = os.path.join(self.root, file_name)
        box = ann['bbox']
        instance = {}
        if 'mask' in self.ann_types:
            if isinstance(ann['segmentation'], list):
                instance['mask'] = ann['segmentation']
            else:
                instance['mask'] = os.path.join(self.root.replace('images', 'masks'), ann['segmentation'])
        if 'keypoints' in self.ann_types:
            instance['keypoints'] = ann['keypoints']
        if 'parsing' in self.ann_types:
            instance['parsing'] = [self.root, file_name, ann['parsing_id']]
        if 'uv' in self.ann_types:
            instance['uv'] = [ann['dp_x'], ann['dp_y'], ann['dp_I'], ann['dp_U'], ann['dp_V'], ann['dp_masks']]
        classes = self.json_category_id_to_contiguous_id[ann["category_id"]]

        img = np.asarray(Image.open(img_path).convert('RGB'))
        target = Instance(box, (img.shape[1], img.shape[0]), classes, self.ann_types, instance)

        # transform
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, img_id):
        img_data = self.coco.imgs[img_id]
        return img_data


class COCOInstanceTestDataset(data.Dataset):
    def __init__(self, ann_file, root, bbox_file, image_thresh, ann_types, transforms=None, extra_fields={}):
        self.root = root
        self.coco = COCO(ann_file)
        self.coco_dt = self.coco.loadRes(bbox_file) if bbox_file else None
        self.image_thresh = image_thresh
        self.ann_types = ann_types
        self.transforms = transforms
        self.ids = sorted(self.coco.imgs.keys())
        self.extra_fields = extra_fields

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        category_ids = sorted(self.coco.getCatIds())
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(category_ids)}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        category_ids = [c['name'] for c in self.coco.loadCats(category_ids)]
        self.classes = category_ids

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        if self.coco_dt:
            ann_ids = self.coco_dt.getAnnIds(imgIds=img_id)
            anno = self.coco_dt.loadAnns(ann_ids)
            anno = [obj for obj in anno if obj['score'] > self.image_thresh]
        else:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anno = self.coco.loadAnns(ann_ids)

        try:
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
        except:
            file_name = self.coco.loadImgs(img_id)[0]['coco_url'].split('.org/')[-1]
        img = np.asarray(Image.open(os.path.join(self.root, file_name)).convert('RGB'))
        # filter crowd annotations
        # TODO might be better to add an extra field
        if len(anno) > 0:
            if 'iscrowd' in anno[0]:
                anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]

        scores = [obj["score"] for obj in anno] if self.coco_dt else None

        target = InstanceBox(boxes, classes, (img.shape[1], img.shape[0]), scores)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, idx):
        img_id = self.id_to_img_map[idx]
        img_data = self.coco.imgs[img_id]
        return img_data

    def pull_image(self, idx):
        """Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            idx (int): index of img to show
        Return:
            img
        """
        img_id = self.id_to_img_map[idx]

        try:
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
        except:
            file_name = self.coco.loadImgs(img_id)[0]['coco_url'].split('.org/')[-1]

        return Image.open(os.path.join(self.root, file_name)).convert('RGB')

import os
import cv2
import json

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision
from pycocotools.coco import COCO

from lib.datasets.structures.bounding_box import BoxList
from lib.datasets.structures.mask import Mask
from lib.datasets.structures.semantic_segmentation import SemanticSegmentation, get_semseg
from lib.datasets.structures.keypoint import PersonKeypoints
from lib.datasets.structures.parsing import Parsing, get_parsing

min_keypoints_per_image = 10
min_hier_per_image = 1


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _count_visible_hier(anno):
    return sum(sum(1 for v in ann["hier"][4::5] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno, ann_types, filter_crowd=True):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    if filter_crowd:
        # if image only has crowd annotation, it should be filtered
        if 'iscrowd' in anno[0]:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    if 'keypoints' in ann_types:
        # keypoints task have a slight different critera for considering
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        keypoints_vis = _count_visible_keypoints(anno) >= min_keypoints_per_image
    else:
        keypoints_vis = True
    if 'hier' in ann_types:
        hier_vis = _count_visible_hier(anno) >= min_hier_per_image
    else:
        hier_vis = True

    if keypoints_vis and hier_vis:
        return True

    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations, ann_types, transforms=None,
            extra_fields={}
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno, ann_types):
                    ids.append(img_id)
            self.ids = ids

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        category_ids = sorted(self.coco.getCatIds())
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(category_ids)}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        categories = [c['name'] for c in self.coco.loadCats(category_ids)]
        self.classes = categories
        self.ann_types = ann_types
        self.extra_fields = extra_fields
        self._transforms = transforms
        BoxList.FLIP_MAP = self.extra_fields['box_flip_map'] if 'box_flip_map' in self.extra_fields else ()
        if 'parsing' in self.ann_types:
            Parsing.FLIP_MAP = self.extra_fields['flip_map'] if 'flip_map' in self.extra_fields else ()
        if 'semseg' in self.ann_types:
            SemanticSegmentation.FLIP_MAP = self.extra_fields['flip_map'] if 'flip_map' in self.extra_fields else ()
            self.extra_seg = None
            if 'seg_json' in self.extra_fields:
                if 'panoptic' in self.ann_types:
                    pano_anns = json.load(open(self.extra_fields['seg_json'], 'r'))
                    self.extra_fields['pano_anns'] = pano_anns
                    self.contiguous_category_id_to_json_pano_id = {
                        i: v['id'] for i, v in enumerate(pano_anns['categories'])
                    }
                    self.json_pano_id_to_contiguous_category_id = {
                        v: k for k, v in self.contiguous_category_id_to_json_pano_id.items()
                    }
                else:
                    if self.extra_fields['semseg_format'] == 'poly':
                        self.extra_seg = COCO(self.extra_fields['seg_json'])
                        seg_category_ids = sorted(self.extra_seg.getCatIds() + category_ids)
                        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(seg_category_ids)}
                        self.contiguous_category_id_to_json_id = {v: k for k, v in
                                                                  self.json_category_id_to_contiguous_id.items()}
                        self.extra_fields["json_category_id"] = sorted(self.extra_seg.getCatIds())
                        self.extra_fields["json_category_id_to_contiguous_id"] = self.json_category_id_to_contiguous_id

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        if len(anno) > 0:
            if 'iscrowd' in anno[0]:
                anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if 'mask' in self.ann_types:
            masks = [obj["segmentation"] for obj in anno]
            masks = Mask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if 'semseg' in self.ann_types:
            if self.extra_fields['semseg_format'] == "mask":
                semsegs = get_semseg(self.root, self.coco.loadImgs(self.ids[idx])[0]['file_name'], self.extra_fields)
                semsegs = SemanticSegmentation(semsegs, classes, img.size, mode='mask', extra_fields=self.extra_fields)
                target.add_field("semsegs", semsegs)
            elif self.extra_fields['semseg_format'] == "poly":
                extra_semsegs = self.extra_seg.loadAnns(self.extra_seg.getAnnIds(self.ids[idx])) if self.extra_seg else None
                semsegs = [[obj["segmentation"] for obj in anno], extra_semsegs]
                semsegs = SemanticSegmentation(semsegs, classes, img.size, mode='poly', extra_fields=self.extra_fields)
                target.add_field("semsegs", semsegs)

        if 'keypoints' in self.ann_types:
            if anno and "keypoints" in anno[0]:
                keypoints = []
                for obj in anno:
                    for i, v in enumerate(obj["keypoints"]):
                        if i % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # Therefore we assume the coordinates are "pixel indices" and
                            # add 0.5 to convert to floating point coordinates.
                            obj["keypoints"][i] = v + 0.5
                    keypoints.append(obj["keypoints"])
                keypoints = PersonKeypoints(keypoints, img.size)
                target.add_field("keypoints", keypoints)

        if 'parsing' in self.ann_types:
            if self.extra_fields['semseg_format'] == "mask":
                parsing_ids = [obj["parsing_id"] for obj in anno]
                parsing = get_parsing(self.root, self.coco.loadImgs(self.ids[idx])[0]['file_name'], parsing_ids)
                parsing = Parsing(parsing, img.size)
                target.add_field("parsing", parsing)
            else:
                raise NotImplementedError

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # We convert format after transform because
        # polygon format is faster than bitmask
        if 'mask' in self.ann_types:
            if self.extra_fields['mask_format'] == "mask":
                target.add_field("masks", target.get_field("masks").convert("mask"))

        return img, target, idx

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

        path = self.coco.loadImgs(img_id)[0]['file_name']

        return Image.open(os.path.join(self.root, path)).convert('RGB')

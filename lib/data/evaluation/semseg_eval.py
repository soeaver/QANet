import cv2
import numpy as np
import os
from tqdm import tqdm

import torch

from lib.data.structures.semantic_segmentation import convert_pano_to_semseg, convert_poly_to_semseg
from lib.utils.misc import logging_rank


class SemSegEvaluator:
    """
    Evaluate semantic segmentation
    """

    def __init__(self, dataset, root_dir, pre_dir, num_classes, gt_dir=None):
        """
        Initialize SemSegEvaluator
        :return: None
        """
        self.pre_dir = pre_dir
        self.dataset = dataset
        self.num_classes = num_classes
        self.extra_fields = dataset.extra_fields
        self.ids = dataset.ids
        if gt_dir is not None:
            self.gt_dir = gt_dir
        else:
            self.gt_dir = self.extra_fields['seg_root'] if 'seg_root' in self.extra_fields \
                else root_dir.replace('img', 'seg')
        self.ignore_label = self.extra_fields['ignore_label'] if 'ignore_label' in self.extra_fields else 255
        self.label_shift = self.extra_fields['label_shift'] if 'label_shift' in self.extra_fields else 0
        self.name_trans = self.extra_fields['name_trans'] if 'name_trans' in self.extra_fields else ['jpg', 'png']
        self.stats = dict()

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.num_classes)
        return np.bincount(
            self.num_classes * a[k].astype(int) + b[k], minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def generate_gt_png(self, i, image_name, size):
        if 'pano_anns' not in self.extra_fields:
            if self.extra_fields['semseg_format'] == "mask":
                gt_png = cv2.imread(os.path.join(self.gt_dir, image_name), 0) + self.label_shift
            else:
                assert self.extra_fields['semseg_format'] == "poly"
                anno = self.dataset.coco.loadAnns(self.dataset.coco.getAnnIds(i))
                classes = [obj["category_id"] for obj in anno]
                classes = [self.dataset.json_category_id_to_contiguous_id[c] for c in classes]
                classes = torch.tensor(classes)
                extra_seg = self.dataset.extra_seg
                extra_semsegs = extra_seg.loadAnns(extra_seg.getAnnIds(i)) if extra_seg else None
                semsegs_anno = [[obj["segmentation"] for obj in anno], extra_semsegs]
                gt = convert_poly_to_semseg((size[1], size[0]), semsegs_anno, classes, 1, self.extra_fields)
                gt_png = gt.numpy()
        else:
            image_path = os.path.join(self.gt_dir, image_name)
            gt = convert_pano_to_semseg(image_path, self.extra_fields, image_name)
            gt_png = gt.numpy()

        return gt_png

    def evaluate(self):
        logging_rank('Evaluating Semantic Segmentation predictions')
        hist = np.zeros((self.num_classes, self.num_classes))
        for i in tqdm(self.ids, desc='Calculating IoU ..'):
            image_name = self.dataset.coco.imgs[i]['file_name'].replace(self.name_trans[0], self.name_trans[1])
            if not (os.path.exists(os.path.join(self.gt_dir, image_name)) and
                    os.path.exists(os.path.join(self.pre_dir, image_name))):
                continue
            pre_png = cv2.imread(os.path.join(self.pre_dir, image_name), 0)
            gt_png = self.generate_gt_png(i, image_name, pre_png.shape)

            assert gt_png.shape == pre_png.shape, '{} VS {}'.format(str(gt_png.shape), str(pre_png.shape))
            gt = gt_png.flatten()
            pre = pre_png.flatten()
            hist += self.fast_hist(gt, pre)

        def mean_iou(overall_h):
            iu = np.diag(overall_h) / (overall_h.sum(1) + overall_h.sum(0) - np.diag(overall_h) + 1e-10)
            return iu, np.nanmean(iu)

        def per_class_acc(overall_h):
            acc = np.diag(overall_h) / (overall_h.sum(1) + 1e-10)
            return np.nanmean(acc)

        def pixel_wise_acc(overall_h):
            return np.diag(overall_h).sum() / overall_h.sum()

        iou, miou = mean_iou(hist)
        mean_acc = per_class_acc(hist)
        pixel_acc = pixel_wise_acc(hist)
        self.stats.update(dict(IoU=iou, mIoU=miou, MeanACC=mean_acc, PixelACC=pixel_acc))

    def accumulate(self, p=None):
        pass

    def summarize(self):
        iStr = ' {:<18} @[area={:>6s}] = {:0.4f}'
        for k, v in self.stats.items():
            if k == 'IoU':
                continue
            logging_rank(iStr.format(k, 'all', v))

    def __str__(self):
        self.summarize()


def semseg_png(score, dataset=None, img_info=None, output_folder=None, semseg=None, target=None):
    semseg_pres_dir = os.path.join(output_folder, 'semseg_pres')
    if not os.path.exists(semseg_pres_dir):
        os.makedirs(semseg_pres_dir)

    im_name = img_info['file_name']
    extra_fields = dataset.extra_fields
    name_trans = extra_fields['name_trans'] if 'name_trans' in extra_fields else ['jpg', 'png']
    save_semseg_pres = os.path.join(semseg_pres_dir, im_name.replace(name_trans[0], name_trans[1]))
    cv2.imwrite(save_semseg_pres, score.astype(np.uint8))

    if target is not None:
        semseg_gt_dir = os.path.join(output_folder, 'semseg_gt')
        label = target.get_field("semsegs").semseg.squeeze(0).numpy()
        if not os.path.exists(semseg_gt_dir):
            os.makedirs(semseg_gt_dir)
        save_semseg_gt = os.path.join(semseg_gt_dir, im_name.replace(name_trans[0], name_trans[1]))
        cv2.imwrite(save_semseg_gt, label.astype(np.uint8))

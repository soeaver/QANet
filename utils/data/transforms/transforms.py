import random
import math
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F

from pet.ssd.utils.box_utils import matrix_iou
from pet.utils.data.structures.boxlist_ops import remove_boxes_by_center, remove_boxes_by_overlap


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size, preprocess_type, scale_ratios,
                 force_test_scale=[-1, -1]):
        assert preprocess_type in ["none", "random_crop"]
        assert not (preprocess_type == "none" and min_size == -1)
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.preprocess_type = preprocess_type
        self.scale_ratios = scale_ratios
        self.force_test_scale = force_test_scale

    def reset_size(self, image_size, based_scale_size):
        if self.min_size != -1:
            h, w = based_scale_size
        else:
            h, w = image_size

        if len(self.scale_ratios) == 1:
            scale_ratio = 1
        else:
            scale_ratio = random.uniform(self.scale_ratios[0], self.scale_ratios[1])

        reset_scale_h = int(h * scale_ratio)
        reset_scale_w = int(w * scale_ratio)
        return (reset_scale_h, reset_scale_w)

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        if -1 not in self.force_test_scale:
            size = tuple(force_test_scale)
        else:
            size = self.get_size(image.size)
            if self.preprocess_type == "random_crop":
                size = self.reset_size(image.size, size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomCrop(object):
    def __init__(self, preprocess_type, crop_sizes, pad_pixel=(0, 0, 0), iou_ths=(0.7, )):
        assert preprocess_type in ["none", "random_crop"]
        self.preprocess_type = preprocess_type
        self.crop_sizes = crop_sizes
        self.pad_pixel = tuple(map(int, map(round, pad_pixel)))
        self.iou_ths = iou_ths

    def get_crop_coordinate(self, image_size):
        w, h = image_size
        crop_h, crop_w = random.choice(self.crop_sizes)

        left_change, up_change = w - crop_w, h - crop_h
        left = random.randint(min(0, left_change), max(0, left_change))
        up = random.randint(min(0, up_change), max(0, up_change))

        crop_region = (left, up, min(w, left + crop_w), min(h, up + crop_h))
        crop_shape = (crop_w, crop_h)
        return crop_region, crop_shape

    def image_crop_with_padding(self, img, crop_region, crop_shape):
        set_left, set_up, right, bottom = crop_region
        crop_left, corp_up = max(set_left, 0), max(set_up, 0)
        crop_region = (crop_left, corp_up, right, bottom)

        img = img.crop(crop_region)
        if img.size != crop_shape:
            pad_img = Image.new('RGB', crop_shape, self.pad_pixel)
            paste_region = (max(0-set_left, 0),
                            max(0-set_up, 0),
                            max(0-set_left, 0)+img.size[0],
                            max(0-set_up, 0)+img.size[1])
            pad_img.paste(img, paste_region)
            return pad_img

        return img

    def targets_crop(self, targets, crop_region, crop_shape):
        set_left, set_up, right, bottom = crop_region
        targets = targets.move((set_left, set_up))
        reset_region = (0, 0, min(right-min(set_left, 0), crop_shape[0])-1,
                        min(bottom-min(set_up, 0), crop_shape[1])-1)

        targets = remove_boxes_by_center(targets, reset_region)
        crop_targets = targets.crop(reset_region)
        iou_th = random.choice(self.iou_ths)
        targets = remove_boxes_by_overlap(targets, crop_targets, iou_th)

        targets = targets.set_size(crop_shape)
        # print(len(targets), targets.get_field('parsing').parsing.shape)
        return targets

    def __call__(self, image, targets):
        if self.preprocess_type == "none":
            return image, targets
        crop_region, crop_shape = self.get_crop_coordinate(image.size)
        out_image = self.image_crop_with_padding(image, crop_region, crop_shape)
        out_targets = self.targets_crop(targets, crop_region, crop_shape)
        # if crop_region don't have instance，random crop again.
        if len(out_targets) == 0:
            return self.__call__(image, targets)
        return out_image, out_targets


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5, left_right=()):
        self.prob = prob
        self.left_right = left_right

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0, self.left_right)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    

class SSD_ToTensor(object):
    def __call__(self, image, target):
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1).copy()
        image = torch.from_numpy(image)
        return image, target


class SSD_Distort(object):
    def convert(self, image):
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp
            
        image = image.copy()
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def __call__(self, image, target):
        image = self.convert(image)
        return image, target


class SSD_Mirror(object):
    def __init__(self, left_right):
        self.left_right = left_right

    def mirror(self, image, target):
        _, width, _ = image.shape
        image = image[:, ::-1]
        target = target.ssd_mirror(width, self.left_right)
        return image, target

    def __call__(self, image, target):
        if random.randrange(2):
            image, target = self.mirror(image, target)
        return image, target


class SSD_Init(object):
    def rgb_to_bgr(self, image):
        image = np.array(image)
        image = image[..., ::-1]
        image = image.copy()
        return image

    def __call__(self, image, target):
        image = self.rgb_to_bgr(image)
        return image, target


class SSD_Resize(object):
    def __init__(self, img_size, is_train):
        self.img_size = img_size
        self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        self.is_train = is_train

    def __call__(self, image, target):

        if self.is_train:
            interp_method = self.interp_methods[random.randrange(5)]
        else:
            interp_method = self.interp_methods[0]
        image = cv2.resize(image, self.img_size, interpolation=interp_method)
        target = target.ssd_resize(self.img_size)
        return image, target


class SSD_Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        return image, target


class SSD_CROP_EXPAND(object):
    def __init__(self, fill, prob):
        self.fill = fill
        self.prob = prob

    def get_ori(self, image, target):

        labels = target.get_field("labels").numpy()
        bboxs = target.bbox.numpy()
        ori_labels = labels.copy()
        ori_image = image.copy()
        ori_bbox = bboxs.copy()
        return ori_labels, ori_image, ori_bbox

    def crop_loop(self, image, target):
        image_array = image.copy()
        boxes = target.bbox.numpy()
        height, width, _ = image_array.shape
        while True:
            mode = random.choice((
                None,
                (0.1, None),
                (0.3, None),
                (0.5, None),
                (0.7, None),
                (0.9, None),
                (None, None),
            ))

            if mode is None:
                return image_array, target

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                scale = random.uniform(0.3, 1.)
                min_ratio = max(0.5, scale * scale)
                max_ratio = min(2, 1. / scale / scale)
                ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
                w = int(scale * ratio * width)
                h = int((scale / ratio) * height)

                l = random.randrange(width - w)
                t = random.randrange(height - h)
                roi = np.array((l, t, l + w, t + h))

                iou = matrix_iou(boxes, roi[np.newaxis])

                if not (min_iou <= iou.min() and iou.max() <= max_iou):
                    continue

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                    .all(axis=1)
                boxes_t = boxes[mask].copy()
                labels = target.get_field("labels").numpy()
                labels_t = labels[mask]

                if len(boxes_t) == 0:
                    continue

                image = image_array[roi[1]:roi[3], roi[0]:roi[2]]
                h_roi, w_roi, _ = image.shape
                target = target.ssd_crop(boxes_t, roi, w_roi, h_roi, labels_t)
                return image, target

    def expand_loop(self, image, target):
        image_array = image.copy()
        height, width, depth = image_array.shape

        for _ in range(50):
            scale = random.uniform(1, 4)

            min_ratio = max(0.5, 1. / scale / scale)
            max_ratio = min(2, scale * scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            ws = scale * ratio
            hs = scale / ratio
            if ws < 1 or hs < 1:
                continue
            w = int(ws * width)
            h = int(hs * height)

            left = random.randint(0, w - width)
            top = random.randint(0, h - height)

            expand_image = np.empty(
                (h, w, depth),
                dtype=image_array.dtype)
            expand_image[:, :] = self.fill
            expand_image[top:top + height, left:left + width] = image_array
            image_array = expand_image

            h_roi, w_roi, _ = image_array.shape
            target = target.ssd_expand(left, top, w_roi, h_roi)
            image = image_array.copy()

            return image, target

    def judge(self, image, target):
        boxes = target.bbox.numpy()
        labels = target.get_field("labels").numpy().copy()
        height, width, _ = image.shape
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0]) * 1.
        b_h = (boxes[:, 3] - boxes[:, 1]) * 1.
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        return boxes_t, labels_t

    def __call__(self, image, target):
        ori_labels, ori_image, ori_bbox = self.get_ori(image, target)
        boxes = target.bbox.numpy()

        if len(boxes) == 0:
            image, target = image, target
        else:
            image, target = self.crop_loop(image, target)

        if random.random() > self.prob:
            image, target = image, target
        else:
            image, target = self.expand_loop(image, target)

        now_bbox, now_labels = self.judge(image, target)

        if len(now_bbox) == 0:
            target = target.ssd_collect(ori_bbox, ori_labels)
            image, target = ori_image, target
        else:
            height, width, _ = image.shape
            now_bbox = now_bbox.copy()
            now_bbox[:, 0::2] *= width
            now_bbox[:, 1::2] *= height
            target = target.ssd_collect(now_bbox, now_labels)
            image, target = image, target

        return image, target

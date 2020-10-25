import cv2
import numpy as np
from PIL import Image
import random

import torch
import torchvision
from torchvision.transforms import functional as F

from lib.data.structures.bounding_box import BoxList
from lib.layers.boxlist_ops import matrix_iou, remove_boxes_by_center, remove_boxes_by_overlap
from lib.ops import roi_align_rotated


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
    def __init__(self, min_size, max_size, scales_sampling='choice', scale_factor=()):
        assert scales_sampling in ["choice", "range", "scale_factor", "force_size"]
        if scales_sampling == "range":
            assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))
        if scales_sampling == "force_size":
            assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for force size".format(len(min_size))
        self.min_size = min_size
        self.max_size = max_size
        self.scales_sampling = scales_sampling
        self.scales = scale_factor

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        if self.scales_sampling == 'choice':
            size = random.choice(self.min_size)
        elif self.scales_sampling == 'range':
            size = random.randint(self.min_size[0], self.min_size[1] + 1)
        elif self.scales_sampling == 'scale_factor':
            scale = random.choice(self.scales)
            return int(h * scale), int(w * scale)
        elif self.scales_sampling == 'force_size':
            return int(self.min_size[0]), int(self.min_size[1])
        else:
            raise NotImplementedError
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = size * h / w
        else:
            oh = size
            ow = size * w / h

        return int(oh + 0.5), int(ow + 0.5)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomCrop(object):
    def __init__(self, crop_sizes, pad_pixel=(0, 0, 0), iou_ths=(0.7,), mode='bgr255', border=-1, cat_max_ratio=1, ignore_label=255):
        self.crop_sizes = crop_sizes
        self.pad_pixel = pad_pixel
        self.iou_ths = iou_ths
        self.border = border  # for CenterNet
        self.cat_max_ratio = cat_max_ratio
        self.ignore_label = ignore_label
        self.mode = mode

    def get_border(self, size):
        i = 1
        size /= 2
        while size <= self.border // i:
            i *= 2
        return self.border // i

    def get_crop_coordinate(self, image_size):
        w, h = image_size
        crop_h, crop_w = random.choice(self.crop_sizes)

        w_border = (crop_w / 2) if self.border < 0 else self.get_border(w)
        h_border = (crop_h / 2) if self.border < 0 else self.get_border(h)
        left = random.randint(*sorted(map(int, (w_border, w - w_border)))) - crop_w // 2
        up = random.randint(*sorted(map(int, (h_border, h - h_border)))) - crop_h // 2

        crop_region = (left, up, min(w, left + crop_w), min(h, up + crop_h))
        crop_shape = (crop_w, crop_h)
        return crop_region, crop_shape

    def image_crop_with_padding(self, img, crop_region, crop_shape):
        set_left, set_up, right, bottom = crop_region
        crop_left, corp_up = max(set_left, 0), max(set_up, 0)
        crop_region = (crop_left, corp_up, right, bottom)

        # RGB255
        if self.mode == "rgb":
            pad_pixel = tuple(np.array(self.pad_pixel) * 255)
            pad_pixel = tuple(map(int, map(round, pad_pixel)))
        elif self.mode == "bgr":
            pad_pixel = tuple(np.array(self.pad_pixel[::-1]) * 255)
            pad_pixel = tuple(map(int, map(round, pad_pixel)))
        elif self.mode == "rgb255":
            pad_pixel = tuple(map(int, map(round, self.pad_pixel)))
        elif self.mode == "bgr255":
            pad_pixel = tuple(map(int, map(round, self.pad_pixel[::-1])))
        else:
            raise ValueError("Unknown image format {}!".format(self.mode))

        img = img.crop(crop_region)
        if img.size != crop_shape:
            pad_img = Image.new('RGB', crop_shape, pad_pixel)
            paste_region = (max(0 - set_left, 0),
                            max(0 - set_up, 0),
                            max(0 - set_left, 0) + img.size[0],
                            max(0 - set_up, 0) + img.size[1])
            pad_img.paste(img, paste_region)
            return pad_img

        return img

    def targets_crop(self, targets, crop_region, crop_shape):
        set_left, set_up, right, bottom = crop_region
        targets = targets.move((set_left, set_up))
        reset_region = (0, 0, min(right - min(set_left, 0), crop_shape[0]) - 1,
                        min(bottom - min(set_up, 0), crop_shape[1]) - 1)

        targets = remove_boxes_by_center(targets, reset_region)
        crop_targets = targets.crop(reset_region)
        iou_th = random.choice(self.iou_ths)
        targets = remove_boxes_by_overlap(targets, crop_targets, iou_th)

        targets = targets.set_size(crop_shape)
        return targets

    def __call__(self, image, targets):
        if not len(self.crop_sizes):
            return image, targets
        has_box = len(targets) > 0
        crop_region, crop_shape = self.get_crop_coordinate(image.size)
        if self.cat_max_ratio < 1:
            # repeat 10 times
            for _ in range(10):
                _target = self.targets_crop(targets, crop_region, crop_shape)
                semseg = _target.get_semseg()
                if semseg is not None:
                    labels, cnt = np.unique(semseg, return_counts=True)
                    cnt = cnt[labels != self.ignore_label]
                    if cnt.any():
                        if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                            break
                    crop_region, crop_shape = self.get_crop_coordinate(image.size)

        out_image = self.image_crop_with_padding(image, crop_region, crop_shape)
        out_targets = self.targets_crop(targets, crop_region, crop_shape)

        # if crop_region don't have instance，random crop again.
        if len(out_targets) == 0 and has_box:
            # TODO
            # The depth of recursion in python is limited. When recursion depth
            # greater than 1000(default), a RecursionError will be raised.
            return self.__call__(image, targets)
        return out_image, out_targets


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class Normalize(object):
    def __init__(self, mean, std, mode="bgr255"):
        self.mean = mean
        self.std = std
        self.mode = mode

    def __call__(self, image, target):
        if self.mode == "rgb":
            image = image[[2, 1, 0]] / 255.
        elif self.mode == "bgr":
            image = image / 255.
        elif self.mode == "rgb255":
            image = image[[2, 1, 0]]
        elif self.mode == "bgr255":
            pass
        else:
            raise ValueError("Unknown image format {}!".format(mode))
        image = F.normalize(image.to(torch.float32), mean=self.mean, std=self.std)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        # BGR255
        image = np.asarray(image)[:, :, ::-1]
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.float32))
        return image, target


class ToPILImage(object):
    def __call__(self, image, target):
        image = image.clip(0, 255).astype(np.uint8, copy=False)
        image = Image.fromarray(image, mode='RGB')
        return image, target


class ToNumpy(object):
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, image, target):
        image = np.asarray(image)
        image = image.astype(np.float32 if self.to_float32 else image.dtype, copy=True)
        return image, target


# ssd transform
class CropAndExpand(object):
    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3, ratio_range=(1, 4), prob=0.5,
                 pad_pixel=(0, 0, 0), mode="bgr255"):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.ratio_range = ratio_range
        self.prob = prob

        assert mode in ("rgb", "bgr", "rgb255", "bgr255"), "Unknown image format {}!".format(mode)
        # to rgb255
        pad_pixel = np.array(pad_pixel)
        if "rgb" not in mode:
            pad_pixel = pad_pixel[::-1]
        if "255" not in mode:
            pad_pixel = pad_pixel * 255
        self.pad_pixel = tuple(pad_pixel)

    def crop(self, image, boxes, labels):
        h, w, c = image.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return image, boxes, labels

            min_iou = mode
            for _ in range(50):
                scale = random.uniform(self.min_crop_size, 1.0)
                min_ratio = max(0.5, scale ** 2)
                max_ratio = 1.0 / min_ratio
                ratio = random.uniform(min_ratio, max_ratio) ** 0.5
                new_w = w * min(scale * ratio, 1.0)
                new_h = h * min(scale / ratio, 1.0)

                left = random.uniform(0.0, w - new_w)
                top = random.uniform(0.0, h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue

                overlaps = matrix_iou(boxes, patch[np.newaxis]).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop image
                # only adjust boxes and instance masks when the gt is not empty
                crop_boxes, crop_labels = boxes, labels
                if len(overlaps) > 0:
                    # adjust boxes
                    center = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = ((center[:, 0] > patch[0]) *
                            (center[:, 1] > patch[1]) *
                            (center[:, 0] < patch[2]) *
                            (center[:, 1] < patch[3]))
                    if not mask.any():
                        continue

                    crop_boxes = crop_boxes[mask]
                    crop_labels = crop_labels[mask]

                    crop_boxes[:, 2:] = crop_boxes[:, 2:].clip(max=patch[2:])
                    crop_boxes[:, :2] = crop_boxes[:, :2].clip(min=patch[:2])
                    crop_boxes -= np.tile(patch[:2], 2).astype(crop_boxes.dtype)

                # judge
                boxes_size = crop_boxes[:, 2:] - crop_boxes[:, :2]
                keep = (boxes_size / (patch[2:] - patch[:2]) > 0.01).all(axis=1)
                if not keep.any():
                    return image, boxes, labels

                crop_image = image[patch[1]:patch[3], patch[0]:patch[2]]

                return crop_image, crop_boxes[keep], crop_labels[keep]

    def expand(self, image, boxes, labels):
        if random.uniform(0, 1) > self.prob:
            return image, boxes, labels

        h, w, c = image.shape
        while True:
            scale = random.uniform(*self.ratio_range)
            min_ratio = max(0.5, scale ** -2)
            max_ratio = 1.0 / min_ratio
            ratio = random.uniform(min_ratio, max_ratio) ** 0.5
            ratio_h, ratio_w = scale / ratio, scale * ratio

            if ratio_h < 1 or ratio_w < 1:
                continue

            expand_h, expand_w = int(h * ratio_h), int(w * ratio_w)

            # judge
            boxes_size = boxes[:, 2:] - boxes[:, :2]
            keep = (boxes_size / np.array((expand_w, expand_h)) > 0.01).all(axis=1)
            if not keep.any():
                return image, boxes, labels

            left = int(random.uniform(0, w * ratio_w - w))
            top = int(random.uniform(0, h * ratio_h - h))

            expand_image = np.full((expand_h, expand_w, c), self.pad_pixel, dtype=image.dtype)
            expand_image[top:top + h, left:left + w] = image

            boxes = boxes + np.tile((left, top), 2).astype(boxes.dtype)

            return expand_image, boxes[keep], labels[keep]

    def __call__(self, image, target):
        boxes = target.bbox.numpy()
        labels = target.get_field("labels").numpy()

        image, boxes, labels = self.crop(image, boxes, labels)
        image, boxes, labels = self.expand(image, boxes, labels)

        target = BoxList(boxes, image.shape[1::-1], mode="xyxy")
        target.add_field('labels', torch.Tensor(labels))

        return image, target


# instance transform
class Convert(object):
    def __init__(self, aspect_ratio, max_size=-1):
        self.aspect_ratio = aspect_ratio
        self.max_size = max_size

    def __call__(self, image, target):
        h, w = image.shape[:2]
        max_original_size = float(max((w, h)))
        if 0 < self.max_size < max_original_size:
            scale_ratio = self.max_size / max_original_size
            image = cv2.resize(image, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
        else:
            scale_ratio = 1.0

        target.convert(self.aspect_ratio, scale_ratio)
        return image, target


class Scale(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, image, target):
        target.scale(self.scale_factor)
        return image, target


class Rotate(object):
    def __init__(self, rotation_factor):
        self.rotation_factor = rotation_factor

    def __call__(self, image, target):
        target.rotate(self.rotation_factor)
        return image, target


class Flip(object):
    def __init__(self, flip):
        self.flip = flip

    def __call__(self, image, target):
        if self.flip and random.random() <= 0.5:
            image = image[:, ::-1]
            target.flip()
        return image, target


class CropAndResizeCV2(object):
    def __init__(self, size, affine_mode='cv2'):
        self.size = size
        self.affine_mode = affine_mode

    def __call__(self, image, target):
        target.crop_and_resize(self.size, self.affine_mode)
        assert self.affine_mode in ['cv2', 'roi_align']
        # for test
        if self.affine_mode == 'cv2':
            if isinstance(target.trans, list):
                image_list = []
                for _trans in target.trans:
                    crop_image = cv2.warpAffine(
                        image,
                        _trans,
                        (int(self.size[0]), int(self.size[1])),
                        flags=cv2.INTER_LINEAR
                    )
                    image_list.append(crop_image)
                if len(image_list) > 0:
                    image_list = np.asarray(image_list).transpose(0, 3, 1, 2)
                    image_list = torch.from_numpy(image_list)
                image = image_list
            else:
                image = cv2.warpAffine(
                    image,
                    target.trans,
                    (int(self.size[0]), int(self.size[1])),
                    flags=cv2.INTER_LINEAR
                )

        return image, target


class CropAndResize(object):
    def __init__(self, size, affine_mode='cv2'):
        self.size = size
        self.affine_mode = affine_mode

    def __call__(self, image, target):
        assert self.affine_mode in ['cv2', 'roi_align']
        if self.affine_mode == 'roi_align':
            bbox = target.bbox[None]
            batch_inds = torch.tensor([0.])[None]
            rois = torch.cat([batch_inds, bbox], dim=1)  # Nx5
            image = roi_align_rotated(image[None], rois, (self.size[1], self.size[0]), 1.0, 0, True)[0]

        return image, target


class HalfBody(object):
    def __init__(self, use_half_body, num_half_body, prob_half_body, upper_body_ids,
                 x_ext_half_body, y_ext_half_body):
        self.use_half_body = use_half_body
        self.num_half_body = num_half_body
        self.prob_half_body = prob_half_body
        self.upper_body_ids = upper_body_ids
        self.x_ext_half_body = x_ext_half_body
        self.y_ext_half_body = y_ext_half_body

    def __call__(self, image, target):
        if self.use_half_body and np.random.rand() <= self.prob_half_body:
            target.half_body(self.num_half_body, self.upper_body_ids, self.x_ext_half_body,
                             self.y_ext_half_body)
        return image, target


class GenerateTarget(object):
    def __init__(self, target_type, sigma, prob_size, size):
        self.target_type = target_type
        self.sigma = sigma
        self.prob_size = prob_size
        self.size = size

    def __call__(self, image, target):
        final_target = target.generate_target(self.target_type, self.sigma, self.prob_size, self.size)
        return image, final_target

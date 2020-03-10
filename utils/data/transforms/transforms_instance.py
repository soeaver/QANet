import cv2
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Box2CS(object):
    def __init__(self, aspect_ratio, pixel_std):
        self.aspect_ratio = aspect_ratio
        self.pixel_std = pixel_std

    def __call__(self, image, target):
        target.box2cs(self.aspect_ratio, self.pixel_std)
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
            image = image[:, ::-1, :]
            target.flip()
        return image, target


class Half_Body(object):
    def __init__(self, half_body, num_keypoints_half_body, prob_half_body, points_num, upper_body_ids,
                 x_ext_half_body, y_ext_half_body, aspect_ratio, pose_pixel_std):
        self.half_body = half_body
        self.num_keypoints_half_body = num_keypoints_half_body
        self.prob_half_body = prob_half_body
        self.points_num = points_num
        self.upper_body_ids = upper_body_ids
        self.x_ext_half_body = x_ext_half_body
        self.y_ext_half_body = y_ext_half_body
        self.aspect_ratio = aspect_ratio
        self.pose_pixel_std = pose_pixel_std

    def __call__(self, image, target):
        if self.half_body and random.random() <= self.prob_half_body:
            target.halfbody(self.num_keypoints_half_body, self.points_num, self.upper_body_ids, self.x_ext_half_body,
                            self.y_ext_half_body, self.aspect_ratio, self.pose_pixel_std)
        return image, target


class Affine(object):
    def __init__(self, train_size):
        self.train_size = train_size

    def __call__(self, image, target):
        target.affine(self.train_size)

        image = cv2.warpAffine(
            image,
            target.trans,
            (int(self.train_size[0]), int(self.train_size[1])),
            flags=cv2.INTER_LINEAR)
        return image, target


class Generate_Target(object):
    def __init__(self, target_type, sigma, heatmap_size, train_size):
        self.target_type = target_type
        self.sigma = sigma
        self.heatmap_size = heatmap_size
        self.train_size = train_size

    def __call__(self, image, target):
        final_target = target.generate_target(self.target_type,
                                              self.sigma,
                                              self.heatmap_size,
                                              self.train_size)
        return image, final_target


class BGR_Normalize(object):
    def __init__(self, mean, std, to_rgb=False):
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb

    def __call__(self, image, target):
        if self.to_rgb:
            image = image[[2, 1, 0]]
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

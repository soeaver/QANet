import cv2
import numpy as np
import os
import pycocotools.mask as mask_utils
import random

import torch

from lib.data.structures.densepose_uv import DensePoseMethods, GetDensePoseMask
from lib.ops import roi_align_rotated

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
TO_REMOVE = 1


class Instance(object):
    def __init__(self, bbox, image_size, labels, ann_types=None, instances=None):
        self.bbox = bbox
        self.size = image_size # (w, h)
        self.labels = labels
        self.ann_types = ann_types
        self.instances = {}
        self.aspect_ratio = 1.0
        self.trans = None
        if 'mask' in self.ann_types:
            self.instances['mask'] = Mask(instances['mask'], self.size)
        if 'keypoints' in self.ann_types:
            self.instances['keypoints'] = HeatMapKeypoints(instances['keypoints'])
        if 'parsing' in self.ann_types:
            self.instances['parsing'] = Parsing(instances['parsing'])
        if 'uv' in self.ann_types:
            self.instances['uv'] = Densepose(instances['uv'], self.size, self.bbox)

    def convert(self, aspect_ratio, scale_ratio):
        """
        (x0, y0, w, h) ==> (xc, yc, w, h, a)
        (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
        """
        x0, y0, w, h = self.bbox[:4]
        xc = x0 + w * 0.5
        yc = y0 + h * 0.5

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        w *= 1.25
        h *= 1.25

        self.bbox = torch.tensor([xc, yc, w, h, 0.])
        self.aspect_ratio = aspect_ratio

        if scale_ratio != 1:
            self.bbox = self.bbox * scale_ratio
        for ann_type in self.ann_types:
            self.instances[ann_type].convert(scale_ratio)

    def half_body(self, num_half_body, upper_body_ids, x_ext_half_body, y_ext_half_body):
        if len(self.ann_types) > 1:
            raise NotImplementedError("half body only support one type now")
        for ann_type in self.ann_types:
            half_body_points = self.instances[ann_type].get_half_body_points(num_half_body, upper_body_ids)
        bbox = half_body_transform(half_body_points, x_ext_half_body,
            y_ext_half_body, self.aspect_ratio)

        if bbox is not None:
            self.bbox = bbox

    def scale(self, scale_factor):
        s = np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
        self.bbox[2:4] *= torch.as_tensor(s)

    def rotate(self, rotation_factor):
        r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
            if random.random() <= 0.6 else 0
        self.bbox[4] = torch.as_tensor(r)

    def flip(self):
        # flip center
        self.bbox[0] = self.size[0] - self.bbox[0] - TO_REMOVE
        for ann_type in self.ann_types:
            self.instances[ann_type].flip(self.size[0])

    def crop_and_resize(self, train_size, affine_mode='cv2'):
        self.trans = get_affine_transform(self.bbox, train_size) if affine_mode == 'cv2' else None
        for ann_type in self.ann_types:
            self.instances[ann_type].crop_and_resize(self.bbox, train_size, self.trans)

    def generate_target(self, target_type, sigma, prob_size, train_size):
        target = {}
        if 'mask' in self.ann_types:
            target['mask'] = self.instances['mask'].mask
            target['labels'] = self.labels

        if 'keypoints' in self.ann_types:
            kp_target, kp_target_weight = self.instances['keypoints'].make_heatmap(
                target_type, sigma, prob_size, train_size
            )
            target['keypoints'] = kp_target
            target['keypoints_weight'] = kp_target_weight

        if 'parsing' in self.ann_types:
            target['parsing'] = self.instances['parsing'].parsing.long()

        if 'uv' in self.ann_types:
            target_uv, target_mask = self.instances['uv'].make_target()
            target['uv'] = target_uv
            target['uv_mask'] = target_mask.long()

        return target

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'type={})'.format(self.type)
        return s


class Mask(object):
    def __init__(self, poly_list, image_size):
        if isinstance(poly_list, list):
            rles = mask_utils.frPyObjects(poly_list, image_size[1], image_size[0])
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle).astype(np.int8)
        else:
            mask = cv2.imread(poly_list, 0)
        self.mask = mask

    def convert(self, scale_ratio):
        if scale_ratio != 1:
            self.mask = cv2.resize(self.mask, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_NEAREST)

    def get_half_body_points(self, num_half_body, upper_body_ids):
        raise NotImplementedError("half body only support mask now")
        selected_joints = []
        return selected_joints

    def flip(self, image_w=None):
        flipped_mask = self.mask[:, ::-1]
        self.mask = flipped_mask

    def crop_and_resize(self, bbox, train_size, trans):
        if trans is None:
            mask = torch.from_numpy(np.ascontiguousarray(self.mask)).to(dtype=torch.float32)
            bbox = bbox[None]
            batch_inds = torch.tensor([0.])[None]
            rois = torch.cat([batch_inds, bbox], dim=1)  # Nx5

            self.mask = roi_align_rotated(
                mask[None, None], rois, (train_size[1], train_size[0]), 1.0, 1, True, "nearest"
            ).squeeze()
        else:
            mask = cv2.warpAffine(
                self.mask,
                trans,
                (int(train_size[0]), int(train_size[1])),
                flags=cv2.INTER_NEAREST
            )
            self.mask = torch.from_numpy(mask).to(dtype=torch.float32)


class HeatMapKeypoints(object):
    """
    This class contains the instance operation.
    """

    def __init__(self, keypoints):
        self.keypoints = keypoints
        self.num_keypoints = int(len(self.keypoints) / 3)
        self.joints = None
        self.joints_vis = None
        self.xy2xyz()

    def xy2xyz(self):
        joints = np.zeros([self.num_keypoints, 3])
        joints_vis = np.zeros([self.num_keypoints, 3])
        for ipt in range(self.num_keypoints):
            joints[ipt, 0] = self.keypoints[ipt * 3 + 0]
            joints[ipt, 1] = self.keypoints[ipt * 3 + 1]
            joints[ipt, 2] = 0
            t_vis = self.keypoints[ipt * 3 + 2]
            if t_vis > 1:
                t_vis = 1
            joints_vis[ipt, 0] = t_vis
            joints_vis[ipt, 1] = t_vis
            joints_vis[ipt, 2] = 0

        self.joints = joints
        self.joints_vis = joints_vis

    def convert(self, scale_ratio):
        if scale_ratio != 1:
            self.joints[:, 0:2] *= scale_ratio

    def get_half_body_points(self, num_half_body, upper_body_ids):
        joints = self.joints[:, :2]
        joints_vis = self.joints_vis[:, 1].reshape((-1, 1))
        total_vis = np.sum(joints_vis[:, 0] > 0)
        selected_joints = []
        if total_vis > num_half_body:
            upper_joints = []
            lower_joints = []
            for joint_id in range(self.num_keypoints):
                if joints_vis[joint_id, 0] > 0:
                    if joint_id in upper_body_ids:
                        upper_joints.append(joints[joint_id])
                    else:
                        lower_joints.append(joints[joint_id])

            if np.random.randn() < 0.5 and len(upper_joints) > 3:
                selected_joints = upper_joints
            else:
                selected_joints = lower_joints if len(lower_joints) > 3 else upper_joints
        if len(selected_joints) < 3:
            selected_joints = []
        return selected_joints

    def flip(self, image_w):
        matched_parts = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

        joints = self.joints
        joints_vis = self.joints_vis
        # Flip horizontal
        joints[:, 0] = image_w - joints[:, 0] - TO_REMOVE

        # Change left-right parts
        for pair in matched_parts:
            joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
            joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()
        joints = joints * joints_vis

        self.joints = joints
        self.joints_vis = joints_vis

    def crop_and_resize(self, bbox, train_size, trans):
        joints = self.joints
        if trans is None:
            joints[:, 0:2] = point_affine(joints[:, 0:2], bbox.numpy(), train_size)
        else:
            for i in range(self.num_keypoints):
                if self.joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        self.joints = joints

    def make_heatmap(self, target_type, sigma, prob_size, train_size):
        target_weight = np.ones((self.num_keypoints, 1), dtype=np.float32)
        target_weight[:, 0] = self.joints_vis[:, 0]

        assert target_type == 'gaussian', 'Only support gaussian map now!'

        if target_type == 'gaussian':
            target = np.zeros((self.num_keypoints, prob_size[1], prob_size[0]), dtype=np.float32)
            tmp_size = sigma * 3

            for joint_id in range(self.num_keypoints):
                feat_stride = (train_size[0] // prob_size[0], train_size[1] // prob_size[1])
                mu_x = int(self.joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(self.joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= prob_size[0] or ul[1] >= prob_size[1] or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], prob_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], prob_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], prob_size[0])
                img_y = max(0, ul[1]), min(br[1], prob_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return torch.from_numpy(target), torch.from_numpy(target_weight)


class Parsing(object):
    FLIP_MAP = ()

    def __init__(self, parsing_list):
        root_dir, file_name, parsing_id = parsing_list
        human_dir = root_dir.replace('Images', 'Human_ids')
        category_dir = root_dir.replace('Images', 'Category_ids')
        file_name = file_name.replace('jpg', 'png')
        human_path = os.path.join(human_dir, file_name)
        category_path = os.path.join(category_dir, file_name)
        human_mask = cv2.imread(human_path, 0)
        category_mask = cv2.imread(category_path, 0)
        parsing = category_mask * (human_mask == parsing_id)
        self.parsing = parsing

    def convert(self, scale_ratio):
        if scale_ratio != 1:
            self.parsing = cv2.resize(self.parsing, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_NEAREST)

    def get_half_body_points(self, num_half_body, upper_body_ids):
        parsing_ids = np.unique(parsing)
        selected_joints = []
        if len(parsing_ids) - 1 > num_half_body:
            upper_joints = []
            lower_joints = []
            for joint_id in parsing_ids:
                if joint_id == 0:
                    continue
                mask = np.where(parsing == joint_id, 1, 0)
                if mask.sum() > 100:
                    if joint_id in upper_body_ids:
                        upper_joints.extend(mask_to_bbox(mask))
                    else:
                        lower_joints.extend(mask_to_bbox(mask))

            if np.random.randn() < 0.5 and len(upper_joints) > 6:
                selected_joints = upper_joints
            else:
                selected_joints = lower_joints if len(lower_joints) > 6 else upper_joints
        if len(selected_joints) < 6:
            selected_joints = []
        return selected_joints

    def flip(self, image_w=None):
        flipped_parsing = self.parsing[:, ::-1]
        for l_r in Parsing.FLIP_MAP:
            left = np.where(flipped_parsing == l_r[0])
            right = np.where(flipped_parsing == l_r[1])
            flipped_parsing[left] = l_r[1]
            flipped_parsing[right] = l_r[0]

        self.parsing = flipped_parsing

    def crop_and_resize(self, bbox, train_size, trans):
        if trans is None:
            parsing = torch.from_numpy(np.ascontiguousarray(self.parsing)).to(dtype=torch.float32)
            bbox = bbox[None]
            batch_inds = torch.tensor([0.])[None]
            rois = torch.cat([batch_inds, bbox], dim=1)  # Nx5

            self.parsing = roi_align_rotated(
                parsing[None, None], rois, (train_size[1], train_size[0]), 1.0, 1, True, "nearest"
            ).squeeze()
        else:
            parsing = cv2.warpAffine(
                self.parsing,
                trans,
                (int(train_size[0]), int(train_size[1])),
                flags=cv2.INTER_NEAREST
            )
            self.parsing = torch.from_numpy(parsing)


class Densepose(object):
    def __init__(self, densepose_list, image_size, bbox):
        dp_x, dp_y, dp_I, dp_U, dp_V, dp_masks = densepose_list

        self.mask = GetDensePoseMask(dp_masks)
        self.dp_I = np.array(dp_I)
        self.dp_U = np.array(dp_U)
        self.dp_V = np.array(dp_V)
        self.dp_x = np.array(dp_x)
        self.dp_y = np.array(dp_y)

        self.image_size = image_size
        self.bbox = bbox

    def convert(self, scale_ratio):
        if scale_ratio != 1:
            self.dp_x *= scale_ratio
            self.dp_y *= scale_ratio
            self.image_size *= scale_ratio
            self.bbox *= scale_ratio

    def get_half_body_points(self, num_half_body, upper_body_ids):
        raise NotImplementedError("half body only support uv now")
        selected_joints = []
        return selected_joints

    def flip(self, image_w=None):
        x1, y1 = self.bbox[0], self.bbox[1]
        x2 = x1 + np.maximum(0., self.bbox[2] - TO_REMOVE)
        y2 = y1 + np.maximum(0., self.bbox[3] - TO_REMOVE)
        x1_f = image_w - x2 - TO_REMOVE
        x2_f = image_w - x1 - TO_REMOVE

        self.bbox = [x1_f, y1, x2_f - x1_f, y2 - y1]

        DP = DensePoseMethods()
        f_I, f_U, f_V, f_x, f_y, f_mask = DP.get_symmetric_densepose(
            self.dp_I, self.dp_U, self.dp_V, self.dp_x, self.dp_y, self.mask
        )
        self.mask = f_mask
        self.dp_I = f_I
        self.dp_U = f_U
        self.dp_V = f_V
        self.dp_x = f_x
        self.dp_y = f_y

    def crop_and_resize(self, bbox, train_size, trans):
        # mask
        x1 = int(self.bbox[0])
        y1 = int(self.bbox[1])
        x2 = int(self.bbox[0] + self.bbox[2])
        y2 = int(self.bbox[1] + self.bbox[3])

        x2 = min([x2, self.image_size[0]])
        y2 = min([y2, self.image_size[1]])

        mask = cv2.resize(self.mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        maskim = np.zeros((self.image_size[1], self.image_size[0]))
        maskim[y1:y2, x1:x2] = mask

        if trans is None:
            mask = torch.as_tensor(maskim).to(dtype=torch.float32)
            _bbox = bbox[None]
            batch_inds = torch.tensor([0.])[None]
            rois = torch.cat([batch_inds, _bbox], dim=1)  # Nx5

            self.mask = roi_align_rotated(
                mask[None, None], rois, (train_size[1], train_size[0]), 1.0, 1, True, "nearest"
            ).squeeze()

            # x y I U V
            Point_x = self.dp_x / 255. * self.bbox[2] + self.bbox[0]
            Point_y = self.dp_y / 255. * self.bbox[3] + self.bbox[1]

            coordinate_new = point_affine(
                np.concatenate((Point_x[:, None], Point_y[:, None]), axis=1),
                bbox.numpy(), train_size
            )
            self.dp_x = coordinate_new[:, 0]
            self.dp_y = coordinate_new[:, 1]
        else:
            mask = cv2.warpAffine(
                maskim,
                trans,
                (int(train_size[0]), int(train_size[1])),
                flags=cv2.INTER_NEAREST
            )
            self.mask = torch.from_numpy(mask)

            # x y I U V
            Point_x = self.dp_x / 255. * self.bbox[2] + self.bbox[0]
            Point_y = self.dp_y / 255. * self.bbox[3] + self.bbox[1]

            coordinate_new = affine_transform([Point_x, Point_y], trans)
            self.dp_x, self.dp_y = coordinate_new

    def make_target(self):
        GT_x = torch.zeros(196, dtype=torch.float32)
        GT_y = torch.zeros(196, dtype=torch.float32)
        GT_I = torch.zeros(196, dtype=torch.float32)
        GT_U = torch.zeros(196, dtype=torch.float32)
        GT_V = torch.zeros(196, dtype=torch.float32)

        GT_x[0:len(self.dp_x)] = torch.as_tensor(self.dp_x).to(dtype=torch.float32)
        GT_y[0:len(self.dp_y)] = torch.as_tensor(self.dp_y).to(dtype=torch.float32)
        GT_I[0:len(self.dp_I)] = torch.as_tensor(self.dp_I).to(dtype=torch.float32)
        GT_U[0:len(self.dp_U)] = torch.as_tensor(self.dp_U).to(dtype=torch.float32)
        GT_V[0:len(self.dp_V)] = torch.as_tensor(self.dp_V).to(dtype=torch.float32)

        return torch.stack((GT_x, GT_y, GT_I, GT_U, GT_V), 0), self.mask


def get_affine_transform(box, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    center = np.array([box[0], box[1]], dtype=np.float32)
    scale = np.array([box[2], box[3]], dtype=np.float32)
    rot = box[4]

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def point_affine(points, bbox, out_size):
    points = np.array(points, dtype=np.float32)
    theta = np.pi * bbox[4] / 180
    cos = np.cos(theta)
    sin = np.sin(theta)

    points[:, 0] = (points[:, 0] - (bbox[0] - bbox[2] / 2)) * out_size[0] / bbox[2]
    points[:, 1] = (points[:, 1] - (bbox[1] - bbox[3] / 2)) * out_size[1] / bbox[3]

    points[:, 0] -= (out_size[0] / 2)
    points[:, 1] -= (out_size[1] / 2)
    x = points[:, 1] * sin + points[:, 0] * cos + out_size[0] / 2
    y = points[:, 1] * cos - points[:, 0] * sin + out_size[1] / 2

    points[:, 0] = x
    points[:, 1] = y

    return points


def half_body_transform(half_body_points, x_ext_half_body, y_ext_half_body, aspect_ratio):
    if len(half_body_points) == 0:
        return None

    selected_joints = np.array(half_body_points, dtype=np.float32)

    left_top = np.amin(selected_joints, axis=0)
    right_bottom = np.amax(selected_joints, axis=0)

    center = (left_top + right_bottom) / 2

    w = right_bottom[0] - left_top[0]
    h = right_bottom[1] - left_top[1]

    rand = np.random.rand()
    w *= (1 + rand * x_ext_half_body)
    rand = np.random.rand()
    h *= (1 + rand * y_ext_half_body)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    bbox = torch.tensor([center[0], center[1], w, h, 0.])

    return bbox


def mask_to_bbox(mask):
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return [[x0, y0], [x1, y1]]

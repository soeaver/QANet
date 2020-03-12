import cv2
import torch
import random
import numpy as np

import pycocotools.mask as mask_utils

import utils.data.evaluation.densepose_methods as dp_utils


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Instance(object):
    def __init__(self, bbox, size, image_id, score, ann_types=None, instances=None):
        self.bbox = bbox
        self.image_size = size
        self.image_id = image_id
        self.score = score
        self.ann_types = ann_types
        self.instances = {}
        if 'mask' in self.ann_types:
            self.instances['mask'] = Mask(instances['mask'], self.image_size)
        if 'keypoints' in self.ann_types:
            self.instances['keypoints'] = HeatMapKeypoints(instances['keypoints'])
        if 'parsing' in self.ann_types:
            self.instances['parsing'] = Parsing(instances['parsing'])
        if 'uv' in self.ann_types:
            self.instances['uv'] = Densepose(instances['uv'], self.image_size, self.bbox)

    def box2cs(self, aspect_ratio, pose_pixel_std):
        x, y, w, h = self.bbox[:4]
        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w * 1.0 / pose_pixel_std, h * 1.0 / pose_pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        self._center = center
        self._scale = scale
        self._rotate = 0

    def scale(self, scale_factor):
        s = self._scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)

        self._scale = s

    def rotate(self, rotation_factor):
        r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
            if random.random() <= 0.6 else 0

        self._rotate = r

    def flip(self):
        # flip center
        c = self._center
        c[0] = self.image_size[1] - c[0] - 1

        self._center = c
        for ann_type in self.ann_types:
            self.instances[ann_type].flip(self.image_size[1])

    def halfbody(self, num_keypoints_half_body, points_num, upper_body_ids,
                 x_ext_half_body, y_ext_half_body, aspect_ratio, pose_pixel_std):
        center = self._center
        scale = self._scale
        if 'mask' in self.ann_types:
            raise NotImplementedError("mask not support halfbody now")
        if 'keypoints' in self.ann_types:
            joints = self.instances['keypoints'].joints[:, :2]
            joints_vis = self.instances['keypoints'].joints_vis[:, 1].reshape((-1, 1))
            total_vis = np.sum(joints_vis[:, 0] > 0)
            if total_vis > num_keypoints_half_body:
                c_half_body, s_half_body = half_body_transform(
                    joints, joints_vis, points_num, upper_body_ids, x_ext_half_body,
                    y_ext_half_body, aspect_ratio, pose_pixel_std)

                if c_half_body is not None and s_half_body is not None:
                    center, scale = c_half_body, s_half_body
        if 'parsing' in self.ann_types:
            raise NotImplementedError("parsing not support halfbody now")
        if 'uv' in self.ann_types:
            raise NotImplementedError("uv not support halfbody now")
        self._center = center
        self._scale = scale

    def affine(self, train_size):
        trans = get_affine_transform(self._center, self._scale,
                                     self._rotate, train_size)
        self.trans = trans
        for ann_type in self.ann_types:
            self.instances[ann_type].affine(self.trans, train_size)

    def generate_target(self, target_type, sigma, heatmap_size, train_size):
        target = {}
        if 'mask' in self.ann_types:
            target['mask'] = torch.from_numpy(self.instances['mask'].segmentation).float()
            target['mask_class'] = torch.tensor(self.instances['mask']._class)

        if 'keypoints' in self.ann_types:
            kp_target, kp_target_weight = self.instances['keypoints'].make_heatmap(
                target_type, sigma, heatmap_size, train_size
            )
            target['keypoints'] = torch.from_numpy(kp_target)
            target['keypoints_weight'] = torch.from_numpy(kp_target_weight)

        if 'parsing' in self.ann_types:
            target['parsing'] = torch.from_numpy(self.instances['parsing'].parsing).long()

        if 'uv' in self.ann_types:
            target_UV, target_mask = self.instances['uv'].make_target()
            target['uv'] = torch.tensor(target_UV)
            target['uv_mask'] = torch.from_numpy(target_mask).long()

        bbox = torch.from_numpy(np.asarray(self.bbox)).double()
        target.update(
            {'image_id': self.image_id,
             'center': self._center,
             'scale': self._scale,
             'score': self.score,
             'bbox': bbox,
        })

        return target

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'image_width={}, '.format(self.image_size[1])
        s += 'image_height={}, '.format(self.image_size[0])
        s += 'type={})'.format(self.type)
        return s


class Mask(object):
    def __init__(self, poly_list, image_size):
        if poly_list[0] == 'mask_temp':
            self.train = False
            self.mask = np.array([0])
        else:
            self.train = True
            rles = mask_utils.frPyObjects(poly_list[0], image_size[0], image_size[1])
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle).astype(np.int8)
            self.mask = mask
        self._class = poly_list[1]

    def flip(self, image_w=None):
        if self.train:
            flipped_mask = self.mask[:, ::-1]
            self.mask = flipped_mask

    def affine(self, trans, train_size):
        if self.train:
            mask = self.mask
            self.mask = cv2.warpAffine(
                mask,
                trans,
                (int(train_size[0]), int(train_size[1])),
                flags=cv2.INTER_NEAREST
            )


class HeatMapKeypoints(object):
    """
    This class contains the instance operation.
    """
    def __init__(self, keypoints):
        if keypoints == 'keypoints_temp':
            self.train = False
        else:
            self.train = True
            self.keypoints = keypoints
            self.num_joints = int(len(self.keypoints) / 3)
            self.xy2xyz()

    def xy2xyz(self):
        joints = np.zeros([self.num_joints, 3])
        joints_vis = np.zeros([self.num_joints, 3])
        for ipt in range(self.num_joints):
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

    def flip(self, image_w):
        if self.train:
            matched_parts = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

            joints = self.joints
            joints_vis = self.joints_vis
            # Flip horizontal
            joints[:, 0] = image_w - joints[:, 0] - 1

            # Change left-right parts
            for pair in matched_parts:
                joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
                joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()
            joints = joints * joints_vis

            self.joints = joints
            self.joints_vis = joints_vis

    def affine(self, trans, train_size=None):
        if self.train:
            joints = self.joints

            for i in range(self.num_joints):
                if self.joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

            self.joints = joints

    def make_heatmap(self, target_type, sigma, heatmap_size, train_size):
        if self.train:
            target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
            target_weight[:, 0] = self.joints_vis[:, 0]

            assert target_type == 'gaussian', 'Only support gaussian map now!'

            if target_type == 'gaussian':
                target = np.zeros((self.num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
                tmp_size = sigma * 3

                for joint_id in range(self.num_joints):
                    feat_stride = (train_size[0] // heatmap_size[0], train_size[1] // heatmap_size[1])
                    mu_x = int(self.joints[joint_id][0] / feat_stride[0] + 0.5)
                    mu_y = int(self.joints[joint_id][1] / feat_stride[1] + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
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
                    g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                    v = target_weight[joint_id]
                    if v > 0.5:
                        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            return target, target_weight
        else:
            return np.array([0]), np.array([0])


class Parsing(object):
    def __init__(self, parsing_path):
        if parsing_path == 'parsing_temp':
            self.train = False
            self.parsing = np.array([0])
        else:
            self.train = True
            parsing = cv2.imread(parsing_path, 0)
            self.parsing = parsing

            if 'CIHP' in parsing_path:
                flip_map = ([14, 15], [16, 17], [18, 19])
            elif 'MHP-v2' in parsing_path:
                flip_map = ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33])
            elif 'ATR' in parsing_path:
                flip_map = ([9, 10], [12, 13], [14, 15])
            elif 'LIP' in parsing_path:
                flip_map = ([14, 15], [16, 17], [18, 19])
            else:
                flip_map = ()
            self.flip_map = flip_map

    def flip(self, image_w=None):
        if self.train:
            flipped_parsing = self.parsing[:, ::-1]
            for l_r in self.flip_map:
                left = np.where(flipped_parsing == l_r[0])
                right = np.where(flipped_parsing == l_r[1])
                flipped_parsing[left] = l_r[1]
                flipped_parsing[right] = l_r[0]

            self.parsing = flipped_parsing

    def affine(self, trans, train_size):
        if self.train:
            parsing = self.parsing
            self.parsing = cv2.warpAffine(
                parsing,
                trans,
                (int(train_size[0]), int(train_size[1])),
                flags=cv2.INTER_NEAREST
            )


class Densepose(object):
    def __init__(self, densepose_list, image_size, bbox):
        if densepose_list == 'uv_temp':
            self.train = False
        else:
            self.train = True
            dp_x, dp_y, dp_I, dp_U, dp_V, dp_masks = densepose_list

            self.mask = GetDensePoseMask(dp_masks)
            self.dp_I = np.array(dp_I)
            self.dp_U = np.array(dp_U)
            self.dp_V = np.array(dp_V)
            self.dp_x = np.array(dp_x)
            self.dp_y = np.array(dp_y)

        self.image_size = image_size
        self.bbox = bbox

    def flip(self, image_w=None):
        if self.train:
            x1, y1 = self.bbox[0], self.bbox[1]
            x2 = x1 + np.maximum(0., self.bbox[2] - 1.)
            y2 = y1 + np.maximum(0., self.bbox[3] - 1.)
            x1_f = image_w - x2 - 1
            x2_f = image_w - x1 - 1

            self.bbox = [x1_f, y1, x2_f-x1_f+1, y2-y1+1]

            DP = dp_utils.DensePoseMethods()
            f_I, f_U, f_V, f_x, f_y, f_mask = DP.get_symmetric_densepose(
                self.dp_I, self.dp_U, self.dp_V, self.dp_x, self.dp_y, self.mask
            )
            self.mask = f_mask
            self.dp_I = f_I
            self.dp_U = f_U
            self.dp_V = f_V
            self.dp_x = f_x
            self.dp_y = f_y

    def affine(self, trans, train_size):
        if self.train:
            # mask
            x1 = int(self.bbox[0])
            y1 = int(self.bbox[1])
            x2 = int(self.bbox[0] + self.bbox[2])
            y2 = int(self.bbox[1] + self.bbox[3])

            x2 = min([x2, self.image_size[1]])
            y2 = min([y2, self.image_size[0]])

            mask = cv2.resize(self.mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

            maskim = np.zeros(self.image_size)
            maskim[y1:y2, x1:x2] = mask

            self.mask = cv2.warpAffine(
                maskim,
                trans,
                (int(train_size[0]), int(train_size[1])),
                flags=cv2.INTER_NEAREST
            )

            # x y I U V
            Point_x = self.dp_x / 255. * self.bbox[2] + self.bbox[0]
            Point_y = self.dp_y / 255. * self.bbox[3] + self.bbox[1]

            coordinate_new = affine_transform([Point_x, Point_y], trans)
            self.dp_x, self.dp_y = coordinate_new

    def make_target(self):
        if self.train:
            GT_x = np.zeros(196, dtype=np.float32)
            GT_y = np.zeros(196, dtype=np.float32)
            GT_I = np.zeros(196, dtype=np.float32)
            GT_U = np.zeros(196, dtype=np.float32)
            GT_V = np.zeros(196, dtype=np.float32)

            GT_x[0:len(self.dp_x)] = self.dp_x
            GT_y[0:len(self.dp_y)] = self.dp_y
            GT_I[0:len(self.dp_I)] = self.dp_I
            GT_U[0:len(self.dp_U)] = self.dp_U
            GT_V[0:len(self.dp_V)] = self.dp_V

            return np.array([GT_x, GT_y, GT_I, GT_U, GT_V]), self.mask
        else:
            return np.array([0]), np.array([0])


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
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


def half_body_transform(joints, joints_vis, points_num, upper_body_ids, x_ext_half_body,
                        y_ext_half_body, aspect_ratio, pose_pixel_std):
    upper_joints = []
    lower_joints = []
    for joint_id in range(points_num):
        if joints_vis[joint_id, 0] > 0:
            if joint_id in upper_body_ids:
                upper_joints.append(joints[joint_id])
            else:
                lower_joints.append(joints[joint_id])

    if np.random.randn() < 0.5 and len(upper_joints) > 3:
        selected_joints = upper_joints
    else:
        selected_joints = lower_joints \
            if len(lower_joints) > 3 else upper_joints

    if len(selected_joints) < 3:
        return None, None

    selected_joints = np.array(selected_joints, dtype=np.float32)

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

    scale = np.array([w * 1.0 / pose_pixel_std, h * 1.0 / pose_pixel_std],
        dtype=np.float32)
    return center, scale


def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256,256])
    for i in range(1,15):
        if Polys[i - 1]:
            current_mask = mask_utils.decode(Polys[i-1])
            MaskGen[current_mask>0] = i
    return MaskGen

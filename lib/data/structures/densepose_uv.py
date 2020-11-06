import cv2
import numpy as np
import os.path as osp
import pycocotools.mask as mask_utils
import scipy.spatial.distance
import torch
from scipy.io import loadmat



# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
UV_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data/coco/annotations'))


class DenseposeUVs(object):
    def __init__(self, dp_uvs, size, flip=False):
        self.dp_uvs = dp_uvs  # dp_xs, dp_ys, dp_Is, dp_Us, dp_Vs, dp_masks

        self.flip = flip
        self.size = size
        self.extra_fields = {}

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT implemented")
        uv = DenseposeUVs(self.dp_uvs, self.size, flip=True)

        return uv

    def move(self, gap):
        uv = DenseposeUVs(self.dp_uvs, self.size, self.flip)

        return uv

    def crop(self, box, gt_boxes):
        if self.flip:
            raise NotImplementedError("Not support crop flipped image{}".format(self.flip))
        w, h = box[2] - box[0], box[3] - box[1]
        xmin, ymin, xmax, ymax = gt_boxes.split(1, dim=-1)
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
        cropped_boxes = np.concatenate(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), axis=-1
        ).astype("int32").tolist()
        gt_boxes = np.array(gt_boxes).astype("int32")

        cropped_dp_uvs = []
        for dp_uv, gt_box, cropped_box in zip(self.dp_uvs, gt_boxes, cropped_boxes):
            cropped_dp_uv = []
            if len(dp_uv):
                box_w = int(cropped_box[2] - cropped_box[0])
                box_h = int(cropped_box[3] - cropped_box[1])
                GT_x = np.array(dp_uv[0]) / 256. * float(gt_box[2] - gt_box[0])
                GT_y = np.array(dp_uv[1]) / 256. * float(gt_box[3] - gt_box[1])
                GT_I = np.array(dp_uv[2])
                GT_U = np.array(dp_uv[3])
                GT_V = np.array(dp_uv[4])

                cropped_left = float(box[0] - gt_box[0])
                cropped_up = float(box[1] - gt_box[1])
                cropped_right = float(box[2] - gt_box[0])
                cropped_bottom = float(box[3] - gt_box[1])
                inds = (GT_x >= cropped_left) & (GT_x <= cropped_right) & \
                       (GT_y >= cropped_up) & (GT_y <= cropped_bottom)
                GT_x = (GT_x[inds] - max(cropped_left, 0)) / box_w * 256.
                GT_y = (GT_y[inds] - max(cropped_up, 0)) / box_h * 256.
                GT_I, GT_U, GT_V = GT_I[inds], GT_U[inds], GT_V[inds]

                Ilabel = GetDensePoseMask(dp_uv[5])
                Ilabel = cv2.resize(Ilabel, (int(gt_box[2]) - int(gt_box[0]), int(gt_box[3]) - int(gt_box[1])),
                                    interpolation=cv2.INTER_NEAREST)
                cropped_mask = np.zeros((box_h, box_w))

                old_left = cropped_box[0] - gt_box[0]
                old_up = cropped_box[1] - gt_box[1]
                old_right = cropped_box[2] - gt_box[0]
                old_bottom = cropped_box[3] - gt_box[1]

                cropped_mask[:int(box_h), :int(box_w)] = \
                    Ilabel[int(old_up):int(old_bottom), int(old_left):int(old_right)]
                cropped_mask = cv2.resize(cropped_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                polys = DensePoseMask2Polys(cropped_mask)

                cropped_dp_uv.append(GT_x.tolist())
                cropped_dp_uv.append(GT_y.tolist())
                cropped_dp_uv.append(GT_I.tolist())
                cropped_dp_uv.append(GT_U.tolist())
                cropped_dp_uv.append(GT_V.tolist())
                cropped_dp_uv.append(polys)
            cropped_dp_uvs.append(cropped_dp_uv)

        uv = DenseposeUVs(cropped_dp_uvs, (w, h))
        return uv

    def set_size(self, size):
        uv = DenseposeUVs(self.dp_uvs, size, self.flip)

        return uv

    def resize(self, size):
        uv = DenseposeUVs(self.dp_uvs, size, self.flip)

        return uv

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        if isinstance(item, (torch.ByteTensor, torch.BoolTensor)):
            if item.sum() == len(item):
                dp_uvs = self.dp_uvs
            else:
                dp_uvs = []
                for i in range(len(self.dp_uvs)):
                    if item[i]:
                        dp_uvs.append(self.dp_uvs[i])
        else:
            dp_uvs = []
            for i in range(len(item)):
                dp_uvs.append(self.dp_uvs[item[i]])

        uv = DenseposeUVs(dp_uvs, self.size, self.flip)

        return uv

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def __len__(self):
        return len(self.dp_uvs)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_uv={}, ".format(len(self.dp_uvs))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        return s


def uv_on_boxes(targets, rois, prob_size):
    if len(rois) == 0:
        return []
    h, w = prob_size
    K = 24
    bboxes = targets.bbox.cpu().numpy()
    uvs = targets.get_field("uv")
    flip = uvs.flip
    dp_uvs = uvs.dp_uvs
    rois = rois.cpu().numpy()
    # The mask
    All_labels = zeros((rois.shape[0], h, w), int32=True)
    # The points
    X_points = zeros((rois.shape[0], 196), int32=False)
    Y_points = zeros((rois.shape[0], 196), int32=False)
    Ind_points = zeros((rois.shape[0], 196), int32=True)
    I_points = zeros((rois.shape[0], 196), int32=True)
    U_points = zeros((rois.shape[0], 196), int32=False)
    V_points = zeros((rois.shape[0], 196), int32=False)

    for i in range(rois.shape[0]):
        GT_x = np.array(dp_uvs[i][0])
        GT_y = np.array(dp_uvs[i][1])
        GT_I = np.array(dp_uvs[i][2])
        GT_U = np.array(dp_uvs[i][3])
        GT_V = np.array(dp_uvs[i][4])
        Ilabel = GetDensePoseMask(dp_uvs[i][5])

        if flip:
            DP = DensePoseMethods()
            GT_I, GT_U, GT_V, GT_x, GT_y, Ilabel = DP.get_symmetric_densepose(GT_I, GT_U, GT_V, GT_x, GT_y, Ilabel)
        roi_fg = rois[i]
        x1 = roi_fg[0]
        x2 = roi_fg[2]
        y1 = roi_fg[1]
        y2 = roi_fg[3]
        roi_gt = bboxes[i]
        x1_source = roi_gt[0]
        x2_source = roi_gt[2]
        y1_source = roi_gt[1]
        y2_source = roi_gt[3]
        x_targets = (np.arange(x1, x2, (x2 - x1) / w) - x1_source) * (256. / (x2_source - x1_source))
        y_targets = (np.arange(y1, y2, (y2 - y1) / h) - y1_source) * (256. / (y2_source - y1_source))
        x_targets = x_targets[0:w]  # Strangely sometimes it can be M+1, so make sure size is OK!
        y_targets = y_targets[0:h]
        [X_targets, Y_targets] = np.meshgrid(x_targets, y_targets)
        New_Index = cv2.remap(Ilabel, X_targets.astype(np.float32), Y_targets.astype(np.float32),
                              interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
        All_L = New_Index
        gt_length_x = x2_source - x1_source
        gt_length_y = y2_source - y1_source
        GT_y = ((GT_y / 256. * gt_length_y) + y1_source - y1) * (h / (y2 - y1))
        GT_x = ((GT_x / 256. * gt_length_x) + x1_source - x1) * (w / (x2 - x1))
        GT_I[GT_y < 0] = 0
        GT_I[GT_y > (h - 1)] = 0
        GT_I[GT_x < 0] = 0
        GT_I[GT_x > (w - 1)] = 0

        points_inside = GT_I > 0
        GT_U = GT_U[points_inside]
        GT_V = GT_V[points_inside]
        GT_x = GT_x[points_inside]
        GT_y = GT_y[points_inside]
        GT_I = GT_I[points_inside]
        X_points[i, 0:len(GT_x)] = GT_x
        Y_points[i, 0:len(GT_y)] = GT_y
        Ind_points[i, 0:len(GT_I)] = i
        I_points[i, 0:len(GT_I)] = GT_I
        U_points[i, 0:len(GT_U)] = GT_U
        V_points[i, 0:len(GT_V)] = GT_V
        All_labels[i, :] = All_L.astype(np.int32)
    U_points = np.tile(U_points, [1, K + 1])
    V_points = np.tile(V_points, [1, K + 1])
    Uv_Weight_Points = zeros(U_points.shape, int32=False)
    for jjj in range(1, K + 1):
        Uv_Weight_Points[:, jjj * I_points.shape[1]: (jjj + 1) * I_points.shape[1]] = (I_points == jjj).astype(
            np.float32)

    return [All_labels, X_points, Y_points, Ind_points, I_points, U_points, V_points, Uv_Weight_Points]


def flip_uv_featuremap(uvs_hf):
    # Invert the predicted soft uv
    uvs_inv = []
    label_index = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
    _index = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
    osp.join(UV_DATA_DIR, 'DensePoseData/UV_data/UV_symmetry_transforms.mat')
    UV_symmetry_filename = osp.join(UV_DATA_DIR, 'DensePoseData/UV_data/UV_symmetry_transforms.mat')
    UV_sym = loadmat(UV_symmetry_filename)

    for i in range(len(uvs_hf)):
        uvs_hf[i] = uvs_hf[i][:, :, :, ::-1]

    uvs_inv.append(uvs_hf[0][:, label_index, :, :])
    uvs_inv.append(uvs_hf[1][:, _index, :, :])

    U_uv, V_uv = uvs_hf[2:]
    U_sym = np.zeros(U_uv.shape)
    V_sym = np.zeros(V_uv.shape)
    U_uv = np.where(U_uv < 0, 0, U_uv)
    V_uv = np.where(V_uv < 0, 0, V_uv)
    U_uv = np.where(U_uv > 1, 1, U_uv)
    V_uv = np.where(V_uv > 1, 1, V_uv)
    U_loc = (U_uv * 255).astype(np.int64)
    V_loc = (V_uv * 255).astype(np.int64)
    for i in range(1, 25):
        for j in range(len(V_sym)):
            V_sym[j, i] = UV_sym['V_transforms'][0, i - 1][V_loc[j, i], U_loc[j, i]]
            U_sym[j, i] = UV_sym['U_transforms'][0, i - 1][V_loc[j, i], U_loc[j, i]]

    uvs_inv.append(U_sym[:, _index, :, :])
    uvs_inv.append(V_sym[:, _index, :, :])

    return uvs_inv


def flip_uv_prob(uv_prob):
    device = uv_prob[0].device
    # Invert the predicted soft uv
    uvs_inv = []
    label_index = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
    _index = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
    osp.join(UV_DATA_DIR, 'DensePoseData/UV_data/UV_symmetry_transforms.mat')
    UV_symmetry_filename = osp.join(UV_DATA_DIR, 'DensePoseData/UV_data/UV_symmetry_transforms.mat')
    UV_sym = loadmat(UV_symmetry_filename)

    for i in range(len(uv_prob)):
        uv_prob[i] = torch.flip(uv_prob[i], dims=(3,))

    uvs_inv.append(uv_prob[0][:, label_index, :, :])
    uvs_inv.append(uv_prob[1][:, _index, :, :])

    U_uv, V_uv = uv_prob[2:]
    U_uv, V_uv = U_uv.cpu().numpy(), V_uv.cpu().numpy()
    U_sym = np.zeros(U_uv.shape)
    V_sym = np.zeros(V_uv.shape)
    U_uv = np.where(U_uv < 0, 0, U_uv)
    V_uv = np.where(V_uv < 0, 0, V_uv)
    U_uv = np.where(U_uv > 1, 1, U_uv)
    V_uv = np.where(V_uv > 1, 1, V_uv)
    U_loc = (U_uv * 255).astype(np.int64)
    V_loc = (V_uv * 255).astype(np.int64)
    for i in range(1, 25):
        for j in range(len(V_sym)):
            V_sym[j, i] = UV_sym['V_transforms'][0, i - 1][V_loc[j, i], U_loc[j, i]]
            U_sym[j, i] = UV_sym['U_transforms'][0, i - 1][V_loc[j, i], U_loc[j, i]]

    uvs_inv.append(torch.from_numpy(U_sym[:, _index, :, :]).to(dtype=torch.float32, device=device))
    uvs_inv.append(torch.from_numpy(V_sym[:, _index, :, :]).to(dtype=torch.float32, device=device))

    return uvs_inv


def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256, 256])
    for i in range(1, 15):
        if Polys[i - 1]:
            current_mask = mask_utils.decode(Polys[i - 1])
            MaskGen[current_mask > 0] = i
    return MaskGen


def DensePoseMask2Polys(MaskGen):
    Polys = []
    for i in range(1, 15):
        current_polys = []
        current_mask = np.zeros([256, 256], dtype='uint8', order='F')
        idx = (MaskGen == i)
        current_mask[idx] = MaskGen[idx]
        if len(idx):
            current_polys = mask_utils.encode(current_mask)
        Polys.append(current_polys)
    return Polys


def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)


class DensePoseMethods:
    def __init__(self):
        assert osp.exists(osp.join(UV_DATA_DIR, 'DensePoseData/UV_data/UV_Processed.mat'))
        assert osp.exists(osp.join(UV_DATA_DIR, 'DensePoseData/UV_data/UV_symmetry_transforms.mat'))

        ALP_UV = loadmat(osp.join(UV_DATA_DIR, 'DensePoseData/UV_data/UV_Processed.mat'))
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]
        # Info to compute symmetries.
        self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.Index_Symmetry_List = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21,
                                    24, 23]
        UV_symmetry_filename = osp.join(UV_DATA_DIR, 'DensePoseData/UV_data/UV_symmetry_transforms.mat')
        self.UV_symmetry_transformations = loadmat(UV_symmetry_filename)

    def get_symmetric_densepose(self, I, U, V, x, y, Mask):
        # This is a function to get the mirror symmetric UV labels.
        Labels_sym = np.zeros(I.shape)
        U_sym = np.zeros(U.shape)
        V_sym = np.zeros(V.shape)
        for i in (range(24)):
            if i + 1 in I:
                Labels_sym[I == (i + 1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i + 1))
                U_loc = (U[jj] * 255).astype(np.int64)
                V_loc = (V[jj] * 255).astype(np.int64)
                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0, i][V_loc, U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0, i][V_loc, U_loc]
        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)

        for i in (range(14)):
            Mask_flipped[Mask_flip == (i + 1)] = self.SemanticMaskSymmetries[i + 1]
        [y_max, x_max] = Mask_flip.shape
        y_sym = y
        x_sym = x_max - x

        return Labels_sym, U_sym, V_sym, x_sym, y_sym, Mask_flipped

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0

        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if np.dot(vCrossW, vCrossU) < 0:
            return False

        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)

        if np.dot(uCrossW, uCrossV) < 0:
            return False

        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom

        return (r <= 1) & (t <= 1) & (r + t <= 1)

    def barycentric_coordinates(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0

        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)

        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom

        return 1 - (r + t), r, t

    def IUV2FBC(self, I_point, U_point, V_point):
        P = [U_point, V_point, 0]
        FaceIndicesNow = np.where(self.FaceIndices == I_point)
        FacesNow = self.FacesDensePose[FaceIndicesNow]
        P_0 = np.vstack((self.U_norm[FacesNow][:, 0], self.V_norm[FacesNow][:, 0],
                         np.zeros(self.U_norm[FacesNow][:, 0].shape))).transpose()
        P_1 = np.vstack((self.U_norm[FacesNow][:, 1], self.V_norm[FacesNow][:, 1],
                         np.zeros(self.U_norm[FacesNow][:, 1].shape))).transpose()
        P_2 = np.vstack((self.U_norm[FacesNow][:, 2], self.V_norm[FacesNow][:, 2],
                         np.zeros(self.U_norm[FacesNow][:, 2].shape))).transpose()

        for i, [P0, P1, P2] in enumerate(zip(P_0, P_1, P_2)):
            if self.barycentric_coordinates_exists(P0, P1, P2, P):
                [bc1, bc2, bc3] = self.barycentric_coordinates(P0, P1, P2, P)
                return FaceIndicesNow[0][i], bc1, bc2, bc3
        # If the found UV is not inside any faces, select the vertex that is closest!
        D1 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()

        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()

        if (minD1 < minD2) & (minD1 < minD3):
            return FaceIndicesNow[0][np.argmin(D1)], 1., 0., 0.
        elif (minD2 < minD1) & (minD2 < minD3):
            return FaceIndicesNow[0][np.argmin(D2)], 0., 1., 0.
        else:
            return FaceIndicesNow[0][np.argmin(D3)], 0., 0., 1.

    def FBC2PointOnSurface(self, FaceIndex, bc1, bc2, bc3, Vertices):
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1
        p = Vertices[Vert_indices[0], :] * bc1 + Vertices[Vert_indices[1], :] * bc2 + Vertices[Vert_indices[2], :] * bc3

        return p

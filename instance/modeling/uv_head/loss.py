import numpy as np

import torch
from torch.nn import functional as F

from models.ops import PoolPointsInterp
from models.ops import smooth_l1_loss_LW
from instance.core.config import cfg


def UV_loss(outputs, target_UV, target_mask):

    x_Ann, x_Index, x_U, x_V = outputs

    device_id = target_mask.get_device()
    dp_x, dp_y, dp_I, dp_U, dp_V = target_UV.cpu().numpy().transpose((1,0,2))

    dp_Ind = []
    for i, x_per_img in enumerate(dp_x):
        dp_Ind.append((x_per_img > 0) * i)

    dp_x = dp_x.reshape((-1, 1))
    dp_y = dp_y.reshape((-1, 1))
    dp_Ind = np.array(dp_Ind).reshape((-1, 1))

    Coordinate_Shapes = np.concatenate((dp_Ind, dp_x, dp_y), axis=1)
    Coordinate_Shapes = torch.from_numpy(Coordinate_Shapes).cuda(device_id).float()

    PPI_op = PoolPointsInterp()
    interp_U = PPI_op(x_U, Coordinate_Shapes)
    interp_V = PPI_op(x_V, Coordinate_Shapes)
    interp_Index_UV = PPI_op(x_Index, Coordinate_Shapes)

    dp_U = np.tile(dp_U, [1, 25])
    dp_V = np.tile(dp_V, [1, 25])
    UV_Weight = np.zeros(dp_U.shape, dtype=np.float32)
    for jjj in range(1, 25):
        UV_Weight[:, jjj * dp_I.shape[1]: (jjj + 1) * dp_I.shape[1]] = (dp_I == jjj).astype(
            np.float32)

    dp_U = dp_U.reshape((-1, 25, 196))
    dp_U = dp_U.transpose((0, 2, 1))
    dp_U = dp_U.reshape((1, 1, -1, 25))
    dp_U = torch.from_numpy(dp_U).cuda(device_id)

    dp_V = dp_V.reshape((-1, 25, 196))
    dp_V = dp_V.transpose((0, 2, 1))
    dp_V = dp_V.reshape((1, 1, -1, 25))
    dp_V = torch.from_numpy(dp_V).cuda(device_id)

    UV_Weight = UV_Weight.reshape((-1, 25, 196))
    UV_Weight = UV_Weight.transpose((0, 2, 1))
    UV_Weight = UV_Weight.reshape((1, 1, -1, 25))
    UV_Weight = torch.from_numpy(UV_Weight).cuda(device_id)

    dp_I = dp_I.reshape(-1).astype('int64')
    dp_I = torch.from_numpy(dp_I).cuda(device_id)

    loss_seg_AnnIndex = F.cross_entropy(x_Ann, target_mask)
    loss_seg_AnnIndex *= cfg.UV.INDEX_WEIGHTS

    loss_IndexUVPoints = F.cross_entropy(interp_Index_UV, dp_I)
    loss_IndexUVPoints *= cfg.UV.PART_WEIGHTS

    loss_Upoints = smooth_l1_loss_LW(interp_U, dp_U,
                                     UV_Weight, UV_Weight)
    loss_Upoints *= cfg.UV.POINT_REGRESSION_WEIGHTS

    loss_Vpoints = smooth_l1_loss_LW(interp_V, dp_V,
                                     UV_Weight, UV_Weight)
    loss_Vpoints *= cfg.UV.POINT_REGRESSION_WEIGHTS

    return loss_seg_AnnIndex, loss_IndexUVPoints, loss_Upoints, loss_Vpoints

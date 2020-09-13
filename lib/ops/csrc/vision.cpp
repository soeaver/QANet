#include <torch/extension.h>
#include "NMS/ml_nms.h"
#include "NMS/soft_nms.h"
#include "NMS/poly_nms.h"
#include "NMS/nms_rotated.h"
#include "NMS/ml_soft_nms.h"
#include "ROIAlign/ROIAlign.h"
#include "ROIAlign/ROIAlignRotated.h"
#include "ROIPool/ROIPool.h"
#include "FocalLoss/SigmoidFocalLoss.h"
#include "PoolPointsInterp/PoolPointsInterp.h"
#include "Deformable/deform_conv.h"
#include "Box_ops/box_iou.h"
#include "Box_ops/box_iou_rotated.h"
#include "Box_ops/box_voting.h"
#include "Box_ops/box_ml_voting.h"

namespace pet {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward", &ROIAlign_forward, "Forward pass for ROIAlign Operator");
  m.def("roi_align_backward", &ROIAlign_backward, "Backward pass for ROIAlign Operator");
  m.def("roi_align_rotated_forward", &ROIAlignRotated_forward, "Forward pass for Rotated ROI-Align Operator");
  m.def("roi_align_rotated_backward", &ROIAlignRotated_backward, "Backward pass for Rotated ROI-Align Operator");
  m.def("roi_pool_forward", &ROIPool_forward, "Forward pass for ROIPool Operator");
  m.def("roi_pool_backward", &ROIPool_backward, "Backward pass for ROIPool Operator");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "Forward pass for SigmoidFocalLoss Operator");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "Backward pass for SigmoidFocalLoss Operator");
  m.def("pool_points_interp_forward", &PoolPointsInterp_forward, "Forward pass for PoolPointsInterp Operator");
  m.def("pool_points_interp_backward", &PoolPointsInterp_backward, "Backward pass for PoolPointsInterp Operator");
  // nms
  m.def("ml_nms", &ml_nms, "Non-maximum Suppression for Multi Label");
  m.def("soft_nms", &soft_nms, "Soft Non-maximum Suppression");
  m.def("poly_nms", &poly_nms, "Poly Non-maximum Suppression");
  m.def("ml_soft_nms", &ml_soft_nms, "Soft Non-maximum Suppression for Multi Label");
  m.def("nms_rotated", &nms_rotated, "Non-maximum Suppression for rotated boxes");
  // dcn-v2
  m.def("deform_conv_forward", &deform_conv_forward, "Forward pass for Deformable Conv Operator");
  m.def("deform_conv_backward_input", &deform_conv_backward_input, "Backward pass for Deformable Conv Input");
  m.def("deform_conv_backward_filter", &deform_conv_backward_filter, "Backward pass for Deformable Conv Filter");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "Forward pass for Modulated Deformable Conv Operator");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "Forward pass for Modulated Deformable Conv Operator");
  // boxes ops
  m.def("box_iou", &box_iou, "IoU for boxes");
  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes");
  m.def("box_voting", &box_voting, "Voting boxes");
  m.def("box_ml_voting", &box_ml_voting, "Voting boxes for Multi Label");
}

} // namespace pet

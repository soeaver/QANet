#pragma once
#include <torch/types.h>
  
namespace pet {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ml_soft_nms_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float threshold,
    const int method,
    const float sigma,
    const float min_score,
    const int topk);

#ifdef WITH_CUDA
// at::Tensor ml_soft_nms_cuda()
#endif

// Interface for Python
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ml_soft_nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float sigma,
    const float iou_threshold,
    const float min_score,
    const int method,
    const int topk) {
  if (dets.is_cuda()) {
#ifdef WITH_CUDA
//   return ml_soft_nms_cuda()
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ml_soft_nms_cpu(
      dets,
      scores,
      labels,
      iou_threshold,
      method,
      sigma,
      min_score,
      topk);
}

} // namespace pet

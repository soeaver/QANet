#pragma once
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>

namespace pet {

#ifdef WITH_CUDA
at::Tensor ml_nms_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold,
    const int topk);
#endif

inline at::Tensor ml_nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold,
    const int topk) {
  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    if (dets.numel() == 0) {
      at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return ml_nms_cuda(
        dets,
        scores,
        labels,
        iou_threshold,
        topk);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implemented");
}

} // namespace pet

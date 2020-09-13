#pragma once
#include <torch/types.h>
// #include <c10/cuda/CUDAGuard.h>

namespace pet {

#ifdef WITH_CUDA
at::Tensor overlaps_cuda(
    const at::Tensor& boxes,
    const at::Tensor& query_boxes);
#endif

inline at::Tensor box_iou(
    const at::Tensor& boxes,
    const at::Tensor& query_boxes) {
  if (boxes.type().is_cuda()) {
#ifdef WITH_CUDA
    // if (dets.numel() == 0) {
    //   at::cuda::CUDAGuard device_guard(dets.device());
    //   return at::empty({0}, dets.options().dtype(at::kLong));
    // }
    return overlaps_cuda(
        boxes,
        query_boxes);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implemented");
}

} // namespace pet

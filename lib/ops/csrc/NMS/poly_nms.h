#pragma once
#include <torch/types.h>
// #include <c10/cuda/CUDAGuard.h>

namespace pet {

#ifdef WITH_CUDA
at::Tensor poly_nms_cuda(
  const at::Tensor& dets,
  float threshold);
#endif

inline at::Tensor poly_nms(
    const at::Tensor& dets,
    const float threshold) {
  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    if (dets.numel() == 0) {
      // at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return poly_nms_cuda(
        dets,
        threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implemented");
}

} // namespace pet

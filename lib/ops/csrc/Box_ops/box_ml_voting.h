#pragma once
#include <torch/types.h>

namespace pet {

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor, at::Tensor> ml_voting_cuda(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const at::Tensor& query_boxes,
    const at::Tensor& query_scores,
    const at::Tensor& query_labels,
    const int scoring_method,
    const float beta,
    const float threshold);
#endif

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> box_ml_voting(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const at::Tensor& query_boxes,
    const at::Tensor& query_scores,
    const at::Tensor& query_labels,
    const int scoring_method,
    const float beta,
    const float threshold) {
  if (boxes.type().is_cuda()) {
#ifdef WITH_CUDA
    return ml_voting_cuda(
        boxes,
        scores,
        labels,
        query_boxes,
        query_scores,
        query_labels,
        scoring_method,
        beta,
        threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implemented");
}

} // namespace pet

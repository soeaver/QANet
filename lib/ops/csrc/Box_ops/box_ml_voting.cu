#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <iostream>
#include <vector>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

namespace {

template <typename T>
__device__ inline float devIoU(
    T const* const a,
    T const* const b) {
  if (a[4] != b[4]) return 0.0;

  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0.0), height = max(bottom - top, (T)0.0);
  T interS = width * height;
  T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return interS / (Sa + Sb - interS);
}

template <typename T>
__global__ void ml_voting_kernel(
    const int n_boxes,
    const int k_boxes,
    const T* dev_boxes,
    const int64_t* dev_labels,
    const T* dev_query_boxes,
    const T* dev_query_scores,
    const int64_t* dev_query_labels,
    T* dev_dets,
    const int scoring_method,
    const float beta,
    const float threshold) {
  const int col_start = blockIdx.x;
  const int row_start = blockIdx.y;

  const int col_size =
      min(k_boxes - col_start * threadsPerBlock, threadsPerBlock);
  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < row_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_labels[threadsPerBlock * row_start + threadIdx.x];
  }

  __shared__ T block_query_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_query_boxes[threadIdx.x * 5 + 0] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_query_boxes[threadIdx.x * 5 + 1] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_query_boxes[threadIdx.x * 5 + 2] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_query_boxes[threadIdx.x * 5 + 3] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
    block_query_boxes[threadIdx.x * 5 + 4] =
        dev_query_labels[threadsPerBlock * col_start + threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const T* weights = dev_query_scores + threadsPerBlock * col_start;
    const T* cur_box = block_boxes + threadIdx.x * 5;
    int start =
        (row_start * threadsPerBlock + threadIdx.x) * k_boxes +
         col_start * threadsPerBlock;
    int offset;
    T iou, weight;
    T* query_box;

    T* dev_x1 = dev_dets + 0;
    T* dev_y1 = dev_dets + 1;
    T* dev_x2 = dev_dets + 2;
    T* dev_y2 = dev_dets + 3;
    T* dev_score_weight = dev_dets + 4;
    T* dev_score = dev_dets + 5;
    T* dev_box_weight = dev_dets + 6;

    for(int i = 0; i < col_size; i++) {
      offset = (start + i) * 7;
      query_box = block_query_boxes + i * 5;
      iou = devIoU<T>(cur_box, query_box);
      if (iou >= threshold) {
        weight = weights[i];
        dev_x1[offset] = query_box[0] * weight;
        dev_y1[offset] = query_box[1] * weight;
        dev_x2[offset] = query_box[2] * weight;
        dev_y2[offset] = query_box[3] * weight;
        dev_score_weight[offset] = (T)1.0;
        switch (scoring_method) {
          case 0 :  // 'ID'
          case 2 :  // 'AVG'
          case 5 :  // 'QUASI_SUM'
            dev_score[offset] = weight;
            break;
          case 1 :  // 'TEMP_AVG'
            if (weight != (T)0.0) {
              dev_score[offset] =
                (T)1.0 /((T)1.0 + powf((T)1.0 / weight - (T)1.0, (T)1.0 / (T)beta));
            } else {
              dev_score[offset] = weight;
            }
            break;
          case 3 :  // 'IOU_AVG'
            dev_score_weight[offset] = iou;
            dev_score[offset] = iou * weight;
            break;
          case 4 :  // 'GENERALIZED_AVG'
            dev_score[offset] = powf(weight, (T)beta);
            break;
        }
        dev_box_weight[offset] = weight;
      } else {
        dev_x1[offset] = (T)0.0;
        dev_y1[offset] = (T)0.0;
        dev_x2[offset] = (T)0.0;
        dev_y2[offset] = (T)0.0;
        dev_score_weight[offset] = (T)0.0;
        dev_score[offset] = (T)0.0;
        dev_box_weight[offset] = (T)0.0;
      }
    }
  }
}

} // namespace

namespace pet {

std::tuple<at::Tensor, at::Tensor, at::Tensor> ml_voting_cuda(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const at::Tensor& query_boxes,
    const at::Tensor& query_scores,
    const at::Tensor& query_labels,
    const int scoring_method,
    const float beta,
    const float threshold) {
  AT_ASSERTM(boxes.is_cuda(), "boxes must be a CUDA tensor");
  AT_ASSERTM(scores.is_cuda(), "scores must be a CUDA tensor");
  AT_ASSERTM(labels.is_cuda(), "labels must be a CUDA tensor");
  AT_ASSERTM(query_boxes.is_cuda(), "query_boxes must be a CUDA tensor");
  AT_ASSERTM(query_scores.is_cuda(), "query_scores must be a CUDA tensor");
  AT_ASSERTM(query_labels.is_cuda(), "query_labels must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(boxes.device());

  int boxes_num = boxes.size(0);
  int query_boxes_num = query_boxes.size(0);

  const int col_blocks = at::cuda::ATenCeilDiv(query_boxes_num, threadsPerBlock);
  const int row_blocks = at::cuda::ATenCeilDiv(boxes_num, threadsPerBlock);

  at::Tensor dev_dets =
      at::empty({boxes_num, query_boxes_num, 7}, boxes.options());

  dim3 blocks(col_blocks, row_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      boxes.scalar_type(), "ml_voting_kernel_cuda", [&] {
        ml_voting_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            boxes_num,
            query_boxes_num,
            boxes.data_ptr<scalar_t>(),
            labels.data_ptr<int64_t>(),
            query_boxes.data_ptr<scalar_t>(),
            query_scores.data_ptr<scalar_t>(),
            query_labels.data_ptr<int64_t>(),
            dev_dets.data_ptr<scalar_t>(),
            scoring_method,
            beta,
            threshold);
  });

  auto dev_sum = dev_dets.sum(1);
  auto num = dev_sum.select(1, 4);
  auto boxes_ws = dev_sum.select(1, 6);

  auto dev_boxes = 
      dev_sum.narrow(1, 0, 4).div(boxes_ws.unsqueeze(-1)).contiguous();

  auto score_sum = dev_sum.select(1, 5);
  auto dev_scores = scores;
  switch (scoring_method) {
    case 0 :  // 'ID'
      break;
    case 1 :  // 'TEMP_AVG'
    case 2 :  // 'AVG'
    case 3 :  // 'IOU_AVG'
      dev_scores = score_sum.div(num);
      break;
    case 4 :  // 'GENERALIZED_AVG'
      dev_scores = score_sum.div(num).pow(1 / beta);
      break;
    case 5 :  // 'QUASI_SUM'
      dev_scores = score_sum.div(num.pow(beta));
      break;
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(dev_boxes, dev_scores, labels);
}

} // namespace pet

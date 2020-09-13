#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <iostream>
#include <vector>

namespace {

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline float devIoU(
    T const* const a,
    T const* const b) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
  T interS = width * height;
  T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return interS / (Sa + Sb - interS);
}

template <typename T>
__global__ void overlaps_kernel(
    const int n_boxes,
    const int k_boxes,
    const T* dev_boxes,
    const T* dev_query_boxes,
    T* overlaps) {
  const int row_start = blockIdx.x;
  const int col_start = blockIdx.y;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(k_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < row_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 3];
  }

  __shared__ T block_query_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_query_boxes[threadIdx.x * 4 + 0] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_query_boxes[threadIdx.x * 4 + 1] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_query_boxes[threadIdx.x * 4 + 2] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_query_boxes[threadIdx.x * 4 + 3] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const T* cur_box = block_boxes + threadIdx.x * 4;
    T* overlaps_offset =
        overlaps +
        (row_start * threadsPerBlock + threadIdx.x) * k_boxes +
        col_start * threadsPerBlock;
    for(int i = 0; i < col_size; i++) {
        overlaps_offset[i] =
          devIoU<T>(cur_box, block_query_boxes + i * 4);
    }
  }
}

} // namespace

namespace pet {

at::Tensor overlaps_cuda(
    const at::Tensor& boxes,
    const at::Tensor& query_boxes) {
  AT_ASSERTM(boxes.is_cuda(), "boxes must be a CUDA tensor");
  AT_ASSERTM(query_boxes.is_cuda(), "query_boxes must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(boxes.device());

  int boxes_num = boxes.size(0);
  int query_boxes_num = query_boxes.size(0);

  const int row_blocks = at::cuda::ATenCeilDiv(boxes_num, threadsPerBlock);
  const int col_blocks = at::cuda::ATenCeilDiv(query_boxes_num, threadsPerBlock);

  at::Tensor overlaps =
      at::empty({boxes_num, query_boxes_num}, boxes.options());

  dim3 blocks(row_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      boxes.scalar_type(), "overlaps_kernel_cuda", [&] {
        overlaps_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            boxes_num,
            query_boxes_num,
            boxes.data_ptr<scalar_t>(),
            query_boxes.data_ptr<scalar_t>(),
            overlaps.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
  return overlaps;
}

} // namespace pet

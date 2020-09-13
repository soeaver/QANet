#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "../Box_ops/poly_iou_utils.h"

using namespace pet;

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__global__ void poly_nms_kernel(
    const int n_polys,
    const float iou_threshold,
    const T* dev_polys,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
      min(n_polys - row_start * threadsPerBlock, threadsPerBlock);
  const int cols_size =
      min(n_polys - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_polys[threadsPerBlock * 9];
  if (threadIdx.x < cols_size) {
    block_polys[threadIdx.x * 9 + 0] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 0];
    block_polys[threadIdx.x * 9 + 1] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 1];
    block_polys[threadIdx.x * 9 + 2] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 2];
    block_polys[threadIdx.x * 9 + 3] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 3];
    block_polys[threadIdx.x * 9 + 4] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 4];
    block_polys[threadIdx.x * 9 + 5] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 5];
    block_polys[threadIdx.x * 9 + 6] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 6];
    block_polys[threadIdx.x * 9 + 7] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 7];
    block_polys[threadIdx.x * 9 + 8] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 8];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_polys + cur_box_idx * 9;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
        start = threadIdx.x + 1;
    }
    for (i = start; i < cols_size; i++) {
      if (single_poly_iou<T>(cur_box, block_polys + i * 9) > iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = at::cuda::ATenCeilDiv(n_polys, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

namespace pet {

// dets is a N x 9 tensor
at::Tensor poly_nms_cuda(
    const at::Tensor& dets,
    float iou_threshold) {
  AT_ASSERTM(dets.type().is_cuda(), "dets must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(dets.device());

  auto scores = dets.select(1, 8);
  auto order_t = std::get<1>(scores.sort(0, /*descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t);

  int dets_num = dets.size(0);

  const int col_blocks = at::cuda::ATenCeilDiv(dets_num, threadsPerBlock);

  at::Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.scalar_type(), "ml_nms_kernel_cuda", [&] {
        poly_nms_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.data_ptr<scalar_t>(),
            (unsigned long long*)mask.data_ptr<int64_t>());
      });

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.index(
    {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
        .to(order_t.device(), keep.scalar_type())});
}

} // namespace pet

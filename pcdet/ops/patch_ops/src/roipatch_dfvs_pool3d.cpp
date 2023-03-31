#include <torch/serialize/tensor.h>
#include <vector>
// #include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "roipatch_dfvs_pool3d_gpu.h"

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


int roilocal_dfvs_pool3d_wrapper_stack_v2(int batch_size, int pts_num, int boxes_num, int dvs_pts_num, int fps_pts_num, 
    int num_boxes_per_patch, int hash_size, float lambda, float delta, at::Tensor xyz_tensor, at::Tensor boxes3d_tensor, 
    at::Tensor point2patch_indices_tensor, at::Tensor patch2box_indices_tensor,
    at::Tensor pooled_pts_num_tensor, at::Tensor pooled_pts_idx_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(boxes3d_tensor);
    CHECK_INPUT(point2patch_indices_tensor);
    CHECK_INPUT(patch2box_indices_tensor);
    CHECK_INPUT(pooled_pts_num_tensor);
    CHECK_INPUT(pooled_pts_idx_tensor);
    
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *boxes3d = boxes3d_tensor.data_ptr<float>();
    const int *point2patch_indices = point2patch_indices_tensor.data_ptr<int>();
    const int *patch2box_indices = patch2box_indices_tensor.data_ptr<int>();
    int *pooled_pts_num = pooled_pts_num_tensor.data_ptr<int>();
    int *pooled_pts_idx = pooled_pts_idx_tensor.data_ptr<int>();

    roilocal_dfvs_pool3d_kernel_launcher_stack_v2(batch_size, pts_num, boxes_num, dvs_pts_num, fps_pts_num,
      num_boxes_per_patch, hash_size, lambda, delta, xyz, boxes3d, point2patch_indices, patch2box_indices, 
      pooled_pts_num, pooled_pts_idx);
    return 1;
}

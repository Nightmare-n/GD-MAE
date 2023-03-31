#include <torch/serialize/tensor.h>
#include <vector>
// #include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "patch_query_gpu.h"

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

int patch_query_wrapper_stack(int batch_size, int boxes_num, int boxes_per_patch_num,
    int range_x, int range_y, float offset_x, float offset_y, float patch_size_x, float patch_size_y,
    at::Tensor boxes3d_tensor, at::Tensor patch_indices_tensor, at::Tensor patch2box_indices_tensor) {
    CHECK_INPUT(boxes3d_tensor);
    CHECK_INPUT(patch_indices_tensor);
    CHECK_INPUT(patch2box_indices_tensor);
    
    const float *boxes3d = boxes3d_tensor.data_ptr<float>();
    const int *patch_indices = patch_indices_tensor.data_ptr<int>();
    int *patch2box_indices = patch2box_indices_tensor.data_ptr<int>();

    patch_query_kernel_launcher_stack(batch_size, boxes_num, boxes_per_patch_num, range_x, range_y, 
        offset_x, offset_y, patch_size_x, patch_size_y, boxes3d, patch_indices, patch2box_indices);
    return 1;
}

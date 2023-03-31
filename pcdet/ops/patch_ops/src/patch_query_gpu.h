#ifndef _STACK_VOXEL_QUERY_GPU_H
#define _STACK_VOXEL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int patch_query_wrapper_stack(int batch_size, int boxes_num, int boxes_per_patch_num,
    int range_x, int range_y, float offset_x, float offset_y, float patch_size_x, float patch_size_y,
    at::Tensor boxes3d_tensor, at::Tensor patch_indices_tensor, at::Tensor patch2box_indices_tensor);

void patch_query_kernel_launcher_stack(int batch_size, int boxes_num, int boxes_per_patch_num,
    int range_x, int range_y, float offset_x, float offset_y, float patch_size_x, float patch_size_y, 
    const float *boxes3d, const int *patch_indices, int *patch2box_indices);


#endif

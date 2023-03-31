#ifndef _STACK_ROIPATCH_DFVS_POOL3D_GPU_H
#define _STACK_ROIPATCH_DFVS_POOL3D_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int roilocal_dfvs_pool3d_wrapper_stack_v2(int batch_size, int pts_num, int boxes_num, int dvs_pts_num, int fps_pts_num, 
    int num_boxes_per_patch, int hash_size, float lambda, float delta, at::Tensor xyz_tensor, at::Tensor boxes3d_tensor, 
    at::Tensor point2patch_indices_tensor, at::Tensor patch2box_indices_tensor,
    at::Tensor pooled_pts_num_tensor, at::Tensor pooled_pts_idx_tensor);

void roilocal_dfvs_pool3d_kernel_launcher_stack_v2(int batch_size, int pts_num, int boxes_num, int dvs_pts_num, int fps_pts_num, 
    int num_boxes_per_patch, int hash_size, float lambda, float delta, const float *xyz, const float *boxes3d,
    const int *point2patch_indices, const int *patch2box_indices, int *pooled_pts_num, int *pooled_pts_idx);

#endif

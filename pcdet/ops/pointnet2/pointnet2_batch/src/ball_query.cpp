/*
batch version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <torch/serialize/tensor.h>
#include <vector>
// #include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query_gpu.h"

// extern THCState *state;

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


int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_cnt_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx_cnt = idx_cnt_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    
    ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx_cnt, idx);
    return 1;
}

int ball_query_dilated_wrapper_fast(int b, int n, int m, float radius_in, float radius_out, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_cnt_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx_cnt = idx_cnt_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();

    ball_query_dilated_kernel_launcher_fast(b, n, m, radius_in, radius_out, nsample, new_xyz, xyz, idx_cnt, idx);
    return 1;
}
#ifndef _SST_OPS_GPU_H
#define _SST_OPS_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ingroup_inds_wrapper(at::Tensor group_inds_tensor, at::Tensor out_inds_tensor);

void ingroup_inds_launcher(int N, int max_group_id, const long *group_inds, long *out_inds);

int group_inner_inds_wrapper(at::Tensor inverse_inds_tensor, at::Tensor group_inds_tensor);

void group_inner_inds_launcher(int N, int M, int K, const long *inverse_inds, long *group_inds);

#endif

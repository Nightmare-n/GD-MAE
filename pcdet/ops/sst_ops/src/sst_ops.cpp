#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "sst_ops_gpu.h"

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

int ingroup_inds_wrapper(at::Tensor group_inds_tensor, at::Tensor out_inds_tensor) {
    CHECK_INPUT(group_inds_tensor);
    CHECK_INPUT(out_inds_tensor);

    int N = group_inds_tensor.size(0);
    int max_group_id = group_inds_tensor.max().item().toLong();

    const long *group_inds = group_inds_tensor.data_ptr<long>();
    long *out_inds = out_inds_tensor.data_ptr<long>();

    ingroup_inds_launcher(N, max_group_id, group_inds, out_inds);
    return 1;
}

int group_inner_inds_wrapper(at::Tensor inverse_inds_tensor, at::Tensor group_inds_tensor) {
    CHECK_INPUT(inverse_inds_tensor);
    CHECK_INPUT(group_inds_tensor);

    int N = inverse_inds_tensor.size(0);
    int M = group_inds_tensor.size(0);
    int K = group_inds_tensor.size(1);

    const long *inverse_inds = inverse_inds_tensor.data_ptr<long>();
    long *group_inds = group_inds_tensor.data_ptr<long>();

    group_inner_inds_launcher(N, M, K, inverse_inds, group_inds);
    return 1;
}

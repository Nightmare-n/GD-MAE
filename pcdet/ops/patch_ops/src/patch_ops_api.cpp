#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "patch_query_gpu.h"
#include "roipatch_dfvs_pool3d_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("patch_query_wrapper", &patch_query_wrapper_stack, "patch_query_wrapper_stack");
    m.def("roilocal_dfvs_pool3d_wrapper_v2", &roilocal_dfvs_pool3d_wrapper_stack_v2, "roilocal_dfvs_pool3d_wrapper_stack_v2");
}

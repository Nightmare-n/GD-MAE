#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "sst_ops_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ingroup_inds_wrapper", &ingroup_inds_wrapper, "ingroup_inds_wrapper");
    m.def("group_inner_inds_wrapper", &group_inner_inds_wrapper, "group_inner_inds_wrapper");
}

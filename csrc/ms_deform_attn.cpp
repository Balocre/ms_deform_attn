/*!
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "cuda/ms_deform_attn_cuda.h"

#include <torch/extension.h>

using namespace ms_deform_attn;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &forward_cuda, "Multiscale Deformable Attention forward");
  m.def("backward", &backward_cuda, "Multiscale Deformable Attention backward");
}

TORCH_LIBRARY(ms_deform_attn, m)
{
  m.def("forward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, int im2col_step) -> Tensor");
  m.def("backward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, Tensor grad_output, int im2col_step) -> Tensor[]");
}

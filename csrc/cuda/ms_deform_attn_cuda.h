/*!
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

namespace ms_deform_attn
{

    at::Tensor forward_cuda(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const int64_t im2col_step);

    std::vector<at::Tensor> backward_cuda(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const at::Tensor &grad_output,
        const int64_t im2col_step);

}

/*!
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once

#include <torch/extension.h>

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

namespace ms_deform_attn
{

    at::Tensor forward(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const int im2col_step)
    {
        if (value.type().is_cuda())
        {
#ifdef WITH_CUDA
            return forward_cuda(
                value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
            AT_ERROR("Not compiled with GPU support");
#endif
        }
        AT_ERROR("Not implemented on the CPU");
    }

    std::vector<at::Tensor> backward(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const at::Tensor &grad_output,
        const int im2col_step)
    {
        if (value.type().is_cuda())
        {
#ifdef WITH_CUDA
            return backward_cuda(
                value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
            AT_ERROR("Not compiled with GPU support");
#endif
        }
        AT_ERROR("Not implemented on the CPU");
    }

}

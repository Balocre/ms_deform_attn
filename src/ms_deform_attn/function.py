# --------------------------------------------------------------------------------------
# Modified from :
#   https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# --------------------------------------------------------------------------------------

__all__ = ["ms_deform_attn"]

import torch
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class MSDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(  # type: ignore
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        output = torch.ops.ms_deform_attn.forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )

        return output

    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def setup_context(ctx, inputs, output):  # type: ignore
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        ) = inputs

        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )

        ctx.im2col_step = im2col_step

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output):  # type: ignore
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc,
            attn_weight,
        ) = ctx.saved_tensors

        grad_value = grad_value_spatial_shapes = grad_value_level_start_index = (
            grad_sampling_loc
        ) = grad_attn_weight = grad_im2col_step = None

        grad_value, grad_sampling_loc, grad_attn_weight = (
            torch.ops.ms_deform_attn.backward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_loc,
                attn_weight,
                grad_output,
                ctx.im2col_step,
            )
        )

        return (
            grad_value,
            grad_value_spatial_shapes,
            grad_value_level_start_index,
            grad_sampling_loc,
            grad_attn_weight,
            grad_im2col_step,
        )


ms_deform_attn = MSDeformAttnFunction.apply

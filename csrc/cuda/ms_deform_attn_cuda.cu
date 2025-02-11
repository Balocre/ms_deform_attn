#include <vector>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ms_deform_attn_cuda.h"
#include "ms_deform_im2col_cuda.cuh"

using namespace ms_deform_attn;

at::Tensor ms_deform_attn::forward_cuda(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int64_t im2col_step)
{
    TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

    const int64_t batch = value.size(0);
    const int64_t spatial_size = value.size(1);
    const int64_t num_heads = value.size(2);
    const int64_t channels = value.size(3); // batch, spatial_size, num_heads, channels = value.shape

    const int64_t num_levels = spatial_shapes.size(0); // [num_levels, 2]

    const int64_t num_query = sampling_loc.size(1);
    const int64_t num_point = sampling_loc.size(4);

    const int64_t im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto output = at::zeros({batch, num_query, num_heads, channels}, value.options()); // 初始化一个output

    const int64_t batch_n = im2col_step_;
    auto output_n = output.view({batch / im2col_step_, batch_n, num_query, num_heads, channels});
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    for (int64_t n = 0; n < batch / im2col_step_; ++n)
    {
        auto columns = output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "_ms_deform_attn_forward_cuda", ([&]
                                                                                         { ms_deformable_im2col_cuda(
                                                                                               at::cuda::getCurrentCUDAStream(),
                                                                                               value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                                                                                               spatial_shapes.data_ptr<int64_t>(),
                                                                                               level_start_index.data_ptr<int64_t>(),
                                                                                               sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                                                                               attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                                                                               batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                                                                               columns.data_ptr<scalar_t>()); }));
    }

    output = output.view({batch, num_query, num_heads * channels});

    return output;
}

std::vector<at::Tensor> ms_deform_attn::backward_cuda(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int64_t im2col_step)
{
    TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    const int64_t batch = value.size(0);
    const int64_t spatial_size = value.size(1);
    const int64_t num_heads = value.size(2);
    const int64_t channels = value.size(3); // batch, spatial_size, num_heads, channels = value.shape

    const int64_t num_levels = spatial_shapes.size(0);

    const int64_t num_query = sampling_loc.size(1);
    const int64_t num_point = sampling_loc.size(4);

    const int64_t im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int64_t batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = grad_output.view({batch / im2col_step_, batch_n, num_query, num_heads, channels});

    for (int64_t n = 0; n < batch / im2col_step_; ++n) // col2im
    {
        auto grad_output_g = grad_output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "_ms_deform_attn_backward_cuda", ([&]
                                                                                          { ms_deformable_col2im_cuda(
                                                                                                at::cuda::getCurrentCUDAStream(),
                                                                                                grad_output_g.data_ptr<scalar_t>(),
                                                                                                value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                                                                                                spatial_shapes.data_ptr<int64_t>(),
                                                                                                level_start_index.data_ptr<int64_t>(),
                                                                                                sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                                                                                attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                                                                                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                                                                                grad_value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                                                                                                grad_sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                                                                                grad_attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size); }));
    }

    return {grad_value, grad_sampling_loc, grad_attn_weight};
}

TORCH_LIBRARY_IMPL(ms_deform_attn, CUDA, m)
{
    m.impl("forward", &forward_cuda);
    m.impl("backward", &backward_cuda);
}

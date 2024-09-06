# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import pytest
import torch
from ms_deform_attn.function import ms_deform_attn
from ms_deform_attn_core import ms_deform_attn_core_pytorch
from torch.autograd import gradcheck

torch.manual_seed(3)


@pytest.fixture
def args():
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )

    S = sum([(H * W).item() for H, W in shapes])
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2

    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()

    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

    im2col_step = 2

    return (
        value,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )


def _test_correctness(args, rtol, atol):

    (
        value,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ) = args

    expected = (
        ms_deform_attn_core_pytorch(
            value,
            shapes,
            sampling_locations,
            attention_weights,
        )
        .detach()
        .cpu()
    )

    result = (
        ms_deform_attn(
            value,
            shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        .detach()
        .cpu()
    )

    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "dtype,rtol,atol", [(torch.double, None, None), (torch.float, 1e-2, 1e-3)]
)
def test_correctness(args, dtype, rtol, atol):
    (
        value,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ) = args

    value = value.to(dtype)
    sampling_locations = sampling_locations.to(dtype)
    attention_weights = attention_weights.to(dtype)

    args = (
        value,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )

    with torch.no_grad():
        _test_correctness(args, rtol, atol)


# test with TestGradient maybe
def _test_gradient_numerical(
    args, value_shape, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True
):
    (
        _,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ) = args

    # N, Len_in, self.n_heads, int(self.ratio * self.d_model) // self.n_heads

    value = torch.rand(value_shape).cuda() * 0.01

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    gradok = gradcheck(
        ms_deform_attn,
        (
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            attention_weights.double(),
            im2col_step,
        ),
    )

    assert gradok


@pytest.mark.parametrize("value_shape", [(1, 5883, 12, 64), (1, 5883, 16, 64)])
def test_gradient_numerical(args, value_shape):
    _test_gradient_numerical(args, value_shape, True, True, True)

__all__ = ["MSDeformAttn"]

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from .function import ms_deform_attn


def _is_power_of_2(n):  # type: ignore
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_levels: int = 4,
        num_heads: int = 8,
        num_points: int = 4,
        proj_ratio: float = 1.0,
    ):
        """Multi-Scale Deformable Attention Module.

        :param hidden_dim:
            hidden dimension of the features
        :param num_levels:
            number of feature levels
        :param num_heads:
            number of attention heads
        :param num_points:
            number of sampling points per attention head per feature level
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"but got {hidden_dim} and {num_heads}"
            )
        _head_dim = hidden_dim // num_heads
        # you'd better set _head_dim to a power of 2
        # which is more efficient in our CUDA implementation
        if not _is_power_of_2(_head_dim):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.proj_ratio = proj_ratio

        # learnable sampling offset and attention weight of the query
        self.sampling_offsets = nn.Linear(
            hidden_dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            hidden_dim, num_heads * num_levels * num_points
        )

        self.value_proj = nn.Linear(hidden_dim, int(hidden_dim * proj_ratio))
        self.output_proj = nn.Linear(int(hidden_dim * proj_ratio), hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight.data, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )

        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param query:
            (B, Length_{query}, C)
        :param reference_points:
            (B, Length_{query}, num_levels, 2), range in [0, 1], top-left (0,0),
            bottom-right (1, 1), including padding area or (N, Length_{query},
            num_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten:
            (B, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes:
            (num_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index:
            (num_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ...,
            H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask:
            (B, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for
            non-padding elements

        :return output:                 (N, Length_{query}, C)
        """  # noqa: W605

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(
            N,
            Len_in,
            self.num_heads,
            int(self.proj_ratio * self.hidden_dim) // self.num_heads,
        )
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got \
                 {reference_points.shape[-1]} instead."
            )

        output = ms_deform_attn(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.output_proj(output)

        return output

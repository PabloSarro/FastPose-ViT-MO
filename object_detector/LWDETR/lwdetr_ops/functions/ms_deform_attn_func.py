# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
ms_deform_attn_func
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F


# TensorRT 10.14 computes GridSample wrong for these shapes (also standalone, in fp32: max |diff|
# ~3 vs PyTorch and onnxruntime, M9 stretch 2026-11-09). For the TensorRT export
# (optimization/bop_det_trt.py) the sampling is done with gathers instead: set
# GATHER_SAMPLER = True before exporting. Same result as grid_sample(bilinear, zeros,
# align_corners=False).
GATHER_SAMPLER = False


def _grid_sample(value, grid):
    return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def bilinear_sample_gather(value, grid):
    """grid_sample(value, grid, bilinear, zeros padding, align_corners=False) with gathers.
    value (N, C, H, W), grid (N, Ho, Wo, 2) in [-1, 1] (x, y) -> (N, C, Ho, Wo)."""
    N, C, H, W = value.shape
    _, Ho, Wo, _ = grid.shape
    x = ((grid[..., 0] + 1) * W - 1) / 2  # pixel coordinates, (N, Ho, Wo)
    y = ((grid[..., 1] + 1) * H - 1) / 2
    x0, y0 = torch.floor(x), torch.floor(y)
    flat = value.reshape(N, C, H * W)
    out = 0
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi, yi = x0 + dx, y0 + dy
        wgt = (1 - (x - xi).abs()) * (1 - (y - yi).abs())  # bilinear weight of this corner
        valid = (xi >= 0) & (xi <= W - 1) & (yi >= 0) & (yi <= H - 1)
        wgt = wgt * valid.to(wgt.dtype)
        idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).long().reshape(N, 1, Ho * Wo)
        vals = torch.gather(flat, 2, idx.expand(N, C, Ho * Wo)).reshape(N, C, Ho, Wo)
        out = out + vals * wgt[:, None]
    return out
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        """MSDeformAttnFunction forward"""
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """MSDeformAttnFunction backward"""
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """ "for debug and test only, need to use cuda version instead"""
    # B, n_heads, head_dim, N
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape

    # Use static shapes list (list of (H, W)) in export path; avoid dynamic checks entirely
    if isinstance(value_spatial_shapes, (list, tuple)):
        hw_list = [(int(H), int(W)) for H, W in value_spatial_shapes]
    else:
        # Fallback for non-export/training paths
        hw_list = [
            (
                int(H.item()) if isinstance(H, torch.Tensor) else int(H),
                int(W.item()) if isinstance(W, torch.Tensor) else int(W),
            )
            for H, W in value_spatial_shapes
        ]

    value_list = value.split([H * W for H, W in hw_list], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(hw_list):
        # B, n_heads, head_dim, H, W
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        # B, Len_q, n_heads, P, 2 -> B, n_heads, Len_q, P, 2 -> B*n_heads, Len_q, P, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # B*n_heads, head_dim, Len_q, P
        sample = bilinear_sample_gather if GATHER_SAMPLER else _grid_sample
        sampling_value_l_ = sample(value_l_, sampling_grid_l_)
        sampling_value_list.append(sampling_value_l_)
    # (B, Len_q, n_heads, L * P) -> (B, n_heads, Len_q, L, P) -> (B*n_heads, 1, Len_q, L*P)
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    # B*n_heads, head_dim, Len_q, L*P
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()

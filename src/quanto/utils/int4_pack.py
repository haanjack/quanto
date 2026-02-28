"""
INT4 Weight Packing Utilities

Packs BF16 weights with scale/zero_point into actual INT4 format.
This reduces model size by ~4x compared to BF16.

INT4 packing: 8 INT4 values packed into 1 INT32
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping


# Quark's packing order for INT4
PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def quantize_to_int4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Quantize BF16 weight to INT4 values.

    Quark per-group format:
    - Weight: [out_features, in_features]
    - Scale: [out_features, num_groups]
    - Zero point: [out_features, num_groups]

    Args:
        weight: [out_features, in_features] BF16 tensor
        scale: [out_features, num_groups] scale tensor
        zero_point: [out_features, num_groups] zero point tensor
        group_size: size of quantization group

    Returns:
        INT4 values in INT8 tensor [out_features, in_features]
    """
    out_features, in_features = weight.shape
    num_groups = scale.shape[1] if scale.shape[0] == out_features else scale.shape[0]

    # Reshape weight to [out_features, num_groups, group_size]
    weight_grouped = weight.view(out_features, num_groups, group_size)

    # Scale shape: [out_features, num_groups]
    # Need to unsqueeze for broadcasting
    scale_expanded = scale.unsqueeze(-1)  # [out_features, num_groups, 1]
    zp_expanded = zero_point.unsqueeze(-1).float()  # [out_features, num_groups, 1]

    # Quantize: q = round(w / scale) + zero_point
    quantized = torch.round(weight_grouped.float() / scale_expanded) + zp_expanded

    # Clamp to INT4 range: -8 to 7
    quantized = quantized.clamp(-8, 7).to(torch.int8)

    # Reshape back to [out_features, in_features]
    return quantized.view(out_features, in_features)


def pack_int4_to_int32(int4_tensor: torch.Tensor, transpose: bool = True) -> torch.Tensor:
    """
    Pack 8 INT4 values into 1 INT32 along the last dimension.

    Args:
        int4_tensor: [out_features, in_features] INT4 values in INT8
        transpose: If True, transpose for Quark's format

    Returns:
        Packed INT32 tensor
        - If transpose=True: [in_features, out_features // 8] (Quark's format)
        - If transpose=False: [out_features, in_features // 8] (HuggingFace format)
    """
    out_features, in_features = int4_tensor.shape

    if transpose:
        # Transpose for Quark's format: [in_features, out_features]
        int4_tensor = int4_tensor.T.contiguous()
        out_features, in_features = in_features, out_features

    # Pad if not divisible by 8
    if in_features % 8 != 0:
        pad_size = 8 - (in_features % 8)
        int4_tensor = torch.nn.functional.pad(int4_tensor, (0, pad_size))
        in_features = int4_tensor.shape[1]

    # Reshape to [out_features, in_features // 8, 8]
    reshaped = int4_tensor.view(out_features, in_features // 8, 8)

    # Pack 8 INT4 values into 1 INT32 using Quark's order
    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32)

    for i, idx in enumerate(PACK_ORDER):
        val = reshaped[:, :, idx]
        # Mask to 4 bits for INT4 and convert to unsigned for packing
        val = val.to(torch.int32) & 0x0F
        packed = packed | (val << (i * 4))

    return packed


def unpack_int32_to_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack INT32 to INT4 values using Quark order.

    Args:
        packed: [X, Y] INT32 tensor where Y contains 8 packed INT4 values

    Returns:
        [X, Y*8] INT8 tensor with unpacked INT4 values
    """
    unpacked = torch.zeros(packed.shape[0], packed.shape[1] * 8, dtype=torch.int8)

    for i in range(8):
        shift = i * 4
        mask = 0x0F << shift
        vals = (packed & mask) >> shift
        # Convert from unsigned to signed INT4
        vals = vals.to(torch.int8)
        vals = torch.where(vals > 7, vals - 16, vals)
        # Put value from position i to index PACK_ORDER[i]
        unpacked[:, PACK_ORDER[i] :: 8] = vals

    return unpacked


def pack_layer_weights(
    layer_weights: Mapping[str, torch.Tensor],
    group_size: int = 128,
    use_quantizer_prefix: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Pack all Linear layer weights in a layer to INT4.

    Args:
        layer_weights: Dict with weight tensors and quantizer params
        group_size: Quantization group size
        use_quantizer_prefix: If True, look for weight_quantizer.scale format

    Returns:
        Dict with .weight.packed (INT32), .weight.scale (BF16), .weight.zero_point (INT32)
    """
    packed_weights = {}

    # Find all weight keys (excluding quantizer params and non-2D tensors)
    weight_keys = [
        k
        for k in layer_weights.keys()
        if k.endswith(".weight") and "_quantizer" not in k and len(layer_weights[k].shape) == 2
    ]

    for weight_key in weight_keys:
        if use_quantizer_prefix:
            scale_key = f"{weight_key}_quantizer.scale"
            zp_key = f"{weight_key}_quantizer.zero_point"
        else:
            scale_key = f"{weight_key}.scale"
            zp_key = f"{weight_key}.zero_point"

        if scale_key not in layer_weights or zp_key not in layer_weights:
            # Not quantized, copy as-is
            packed_weights[weight_key] = layer_weights[weight_key]
            continue

        weight = layer_weights[weight_key]
        scale = layer_weights[scale_key]
        zero_point = layer_weights[zp_key]

        # Quantize to INT4
        int4_values = quantize_to_int4(weight, scale, zero_point, group_size)

        # Pack to INT32
        packed = pack_int4_to_int32(int4_values)

        # Save packed weight and params
        packed_weights[f"{weight_key}.packed"] = packed
        packed_weights[f"{weight_key}.scale"] = scale
        packed_weights[f"{weight_key}.zero_point"] = zero_point

    # Copy non-Linear weights (norms, etc.)
    for key, tensor in layer_weights.items():
        if "_quantizer" not in key and key not in weight_keys:
            packed_weights[key] = tensor

    return packed_weights

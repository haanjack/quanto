"""
INT4 Weight Packer

Packs BF16 weights with scale/zero_point into actual INT4 format.
This reduces model size by ~4x compared to BF16.

INT4 packing: 8 INT4 values packed into 1 INT32
"""

import torch
from typing import Dict, Tuple


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


def pack_int4_to_int32(int4_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack 8 INT4 values into 1 INT32 along the last dimension.

    Quark's per-group format:
    - Weight is [out_features, in_features]
    - Transpose to [in_features, out_features] for packing
    - Pack 8 out_features into 1 INT32

    Args:
        int4_tensor: [out_features, in_features] INT4 values in INT8

    Returns:
        Packed INT32 tensor [in_features, out_features // 8]
    """
    out_features, in_features = int4_tensor.shape

    # Transpose for Quark's format: [in_features, out_features]
    int4_tensor = int4_tensor.T.contiguous()

    # Pad if not divisible by 8
    if out_features % 8 != 0:
        pad_size = 8 - (out_features % 8)
        int4_tensor = torch.nn.functional.pad(int4_tensor, (0, pad_size))
        out_features = int4_tensor.shape[1]

    # Reshape to [in_features, out_features // 8, 8]
    reshaped = int4_tensor.view(in_features, out_features // 8, 8)

    # Pack 8 INT4 values into 1 INT32 using Quark's order: [0,2,4,6,1,3,5,7]
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros(in_features, out_features // 8, dtype=torch.int32)

    for i, idx in enumerate(order_map):
        val = reshaped[:, :, idx]
        # Mask to 4 bits for INT4
        val = val & 0x0F
        # Convert to unsigned for packing
        val = val.to(torch.int32) & 0x0F
        packed = packed | (val << (i * 4))

    return packed


def pack_layer_weights(
    layer_weights: Dict[str, torch.Tensor],
    group_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Pack all Linear layer weights in a layer to INT4.

    Args:
        layer_weights: Dict with .weight, .weight_quantizer.scale, .weight_quantizer.zero_point
        group_size: Quantization group size

    Returns:
        Dict with .weight.packed (INT32), .weight.scale (BF16), .weight.zero_point (INT32)
    """
    packed_weights = {}

    # Find all weight keys (excluding quantizer params and non-2D tensors)
    weight_keys = [k for k in layer_weights.keys()
                   if k.endswith('.weight') and '_quantizer' not in k and len(layer_weights[k].shape) == 2]

    for weight_key in weight_keys:
        scale_key = f"{weight_key}_quantizer.scale"
        zp_key = f"{weight_key}_quantizer.zero_point"

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
        if '_quantizer' not in key and key not in weight_keys:
            packed_weights[key] = tensor

    return packed_weights


def calculate_compression_ratio(original_size: int, packed_size: int) -> float:
    """Calculate compression ratio."""
    return original_size / packed_size


if __name__ == "__main__":
    # Test packing
    print("Testing INT4 packing...")

    # Create test weight in Quark's format: [out_features, in_features]
    out_features = 1024
    in_features = 4096
    group_size = 128
    num_groups = in_features // group_size

    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    # Quark's scale format: [out_features, num_groups]
    scale = torch.ones(out_features, num_groups, dtype=torch.bfloat16) * 0.1
    zero_point = torch.zeros(out_features, num_groups, dtype=torch.int32)

    print(f"Original weight size: {weight.numel() * 2 / 1024**2:.2f} MB (BF16)")
    print(f"Weight shape: {list(weight.shape)}")
    print(f"Scale shape: {list(scale.shape)}")

    # Quantize
    int4_vals = quantize_to_int4(weight, scale, zero_point, group_size)
    print(f"INT4 values shape: {list(int4_vals.shape)}")

    # Pack
    packed = pack_int4_to_int32(int4_vals)
    print(f"Packed shape: {list(packed.shape)}")
    print(f"Packed size: {packed.numel() * 4 / 1024**2:.2f} MB (INT32)")

    # Compression ratio
    original_mb = weight.numel() * 2  # BF16 = 2 bytes
    packed_mb = packed.numel() * 4    # INT32 = 4 bytes
    print(f"Compression: {original_mb / packed_mb:.2f}x")

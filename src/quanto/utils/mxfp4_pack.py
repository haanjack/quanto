"""
MXFP4 Weight Packing Utilities

Packs BF16 weights into OCP MX FP4 format with E8M0 shared scales.
This reduces model size by ~3.76x compared to BF16.

MXFP4 format (OCP MX Specification):
- FP4 (E2M1): 1 sign + 1 exponent + 2 mantissa = 4 bits per value
- E8M0 scale: 8-bit shared exponent per group of 32 elements
- 2 FP4 values packed per uint8 byte

FP4 E2M1 representable values (magnitude):
  0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
"""

from __future__ import annotations

import math

import torch


# FP4 E2M1 encoding: 4-bit code -> float value
# Bit layout: [sign(1)] [exponent(1)] [mantissa(2)]
# Code 0b_s_e_mm
FP4_E2M1_TABLE = torch.tensor([
    0.0,    # 0b0_0_00 = 0
    0.5,    # 0b0_0_01 = 0.5
    1.0,    # 0b0_0_10 = 1.0
    1.5,    # 0b0_0_11 = 1.5
    2.0,    # 0b0_1_00 = 2.0
    3.0,    # 0b0_1_01 = 3.0
    4.0,    # 0b0_1_10 = 4.0
    6.0,    # 0b0_1_11 = 6.0
    -0.0,   # 0b1_0_00 = -0 (treated as 0)
    -0.5,   # 0b1_0_01 = -0.5
    -1.0,   # 0b1_0_10 = -1.0
    -1.5,   # 0b1_0_11 = -1.5
    -2.0,   # 0b1_1_00 = -2.0
    -3.0,   # 0b1_1_01 = -3.0
    -4.0,   # 0b1_1_10 = -4.0
    -6.0,   # 0b1_1_11 = -6.0
], dtype=torch.float32)

# Max representable magnitude in FP4 E2M1
FP4_MAX = 6.0


def compute_e8m0_scales(weight: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Compute E8M0 shared scales for MXFP4 quantization.

    E8M0 format: 8-bit exponent only (no mantissa), representing power-of-2 scales.
    scale = 2^(e8m0_code - 127)  (IEEE 754 bias)

    The scale is chosen so that max(abs(group)) / scale <= FP4_MAX (6.0).

    Args:
        weight: [..., in_features] tensor to compute scales for
        group_size: Number of elements per group (default 32)

    Returns:
        E8M0 scale codes as uint8 tensor [..., num_groups]
    """
    orig_shape = weight.shape
    in_features = orig_shape[-1]

    # Pad if not divisible by group_size
    if in_features % group_size != 0:
        pad_size = group_size - (in_features % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad_size))
        in_features = weight.shape[-1]

    num_groups = in_features // group_size

    # Reshape to [..., num_groups, group_size]
    grouped = weight.reshape(*orig_shape[:-1], num_groups, group_size)

    # Max absolute value per group
    group_max = grouped.abs().amax(dim=-1).float()

    # Avoid log2(0) — use a small floor
    group_max = group_max.clamp(min=1e-12)

    # E8M0 exponent: floor(log2(group_max / FP4_MAX)) + 127 (IEEE bias)
    # scale = 2^(code - 127), so code = floor(log2(group_max / FP4_MAX)) + 127
    exponent = torch.floor(torch.log2(group_max / FP4_MAX)).to(torch.int32) + 127

    # Clamp to valid E8M0 range [0, 254] (255 = NaN/Inf in E8M0)
    exponent = exponent.clamp(0, 254)

    return exponent.to(torch.uint8)


def e8m0_to_float(e8m0_codes: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 scale codes to float scale values.

    Args:
        e8m0_codes: uint8 tensor of E8M0 exponent codes

    Returns:
        Float tensor of scale values: 2^(code - 127)
    """
    return torch.pow(2.0, e8m0_codes.to(torch.float32) - 127.0)


def quantize_to_fp4(
    weight: torch.Tensor,
    e8m0_scales: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Quantize BF16 weight to FP4 E2M1 codes using E8M0 scales.

    Args:
        weight: [..., in_features] BF16/FP32 tensor
        e8m0_scales: [..., num_groups] uint8 E8M0 scale codes
        group_size: Elements per group

    Returns:
        [..., in_features] uint8 tensor with FP4 codes (values 0-15)
    """
    orig_shape = weight.shape
    in_features = orig_shape[-1]

    # Pad if needed
    if in_features % group_size != 0:
        pad_size = group_size - (in_features % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad_size))
        in_features = weight.shape[-1]

    num_groups = in_features // group_size

    # Convert scales to float
    scales = e8m0_to_float(e8m0_scales)  # [..., num_groups]

    # Reshape weight to [..., num_groups, group_size]
    grouped = weight.reshape(*orig_shape[:-1], num_groups, group_size).float()

    # Divide by scale: normalized = weight / scale
    # scales shape [..., num_groups] -> [..., num_groups, 1]
    normalized = grouped / scales.unsqueeze(-1)

    # Round to nearest FP4 value using the lookup table
    # FP4 positive magnitudes: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    sign = normalized.sign()
    magnitude = normalized.abs()

    # Magnitude-only FP4 table for nearest-value lookup
    fp4_magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=weight.device,
    )

    # Find nearest FP4 magnitude for each value
    # Compute distance to each FP4 magnitude
    diffs = (magnitude.unsqueeze(-1) - fp4_magnitudes).abs()
    mag_codes = diffs.argmin(dim=-1)  # index 0-7

    # Combine sign and magnitude into 4-bit code
    # sign bit = 0 for positive, 1 for negative (bit 3)
    sign_bit = (sign < 0).to(torch.uint8) << 3
    fp4_codes = sign_bit | mag_codes.to(torch.uint8)

    # Reshape back to [..., in_features]
    fp4_codes = fp4_codes.reshape(*orig_shape[:-1], in_features)

    # Trim padding
    if in_features != orig_shape[-1]:
        fp4_codes = fp4_codes[..., : orig_shape[-1]]

    return fp4_codes


def pack_fp4_to_uint8(fp4_codes: torch.Tensor) -> torch.Tensor:
    """
    Pack two FP4 (4-bit) values into one uint8 byte.

    Packing order: low nibble first, high nibble second.
    byte = fp4_codes[2i] | (fp4_codes[2i+1] << 4)

    Args:
        fp4_codes: [..., N] uint8 tensor with FP4 codes (0-15)

    Returns:
        [..., N//2] uint8 tensor with packed FP4 pairs
    """
    *batch, n = fp4_codes.shape

    # Pad if odd number of elements
    if n % 2 != 0:
        fp4_codes = torch.nn.functional.pad(fp4_codes, (0, 1))
        n = fp4_codes.shape[-1]

    # Reshape to [..., N//2, 2]
    pairs = fp4_codes.reshape(*batch, n // 2, 2)

    # Pack: low nibble = first value, high nibble = second value
    packed = (pairs[..., 0] & 0x0F) | ((pairs[..., 1] & 0x0F) << 4)

    return packed.to(torch.uint8)


def unpack_uint8_to_fp4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 bytes to two FP4 codes each.

    Args:
        packed: [..., N] uint8 tensor

    Returns:
        [..., N*2] uint8 tensor with FP4 codes (0-15)
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    # Interleave: [low0, high0, low1, high1, ...]
    unpacked = torch.stack([low, high], dim=-1)
    return unpacked.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def pack_mxfp4(
    weight: torch.Tensor,
    group_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Full pipeline: BF16 weight -> packed MXFP4.

    Args:
        weight: [out_features, in_features] BF16 tensor
        group_size: MXFP4 group size (default 32)

    Returns:
        Dict with:
        - "weight.packed": [out_features, in_features // 2] uint8 (packed FP4 pairs)
        - "weight.scale_e8m0": [out_features, in_features // group_size] uint8
    """
    # Step 1: Compute E8M0 scales
    e8m0_scales = compute_e8m0_scales(weight, group_size)

    # Step 2: Quantize to FP4 codes
    fp4_codes = quantize_to_fp4(weight, e8m0_scales, group_size)

    # Step 3: Pack FP4 codes into uint8
    packed = pack_fp4_to_uint8(fp4_codes)

    return {
        "weight.packed": packed,
        "weight.scale_e8m0": e8m0_scales,
    }


def unpack_mxfp4(
    packed: torch.Tensor,
    scale_e8m0: torch.Tensor,
    group_size: int = 32,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Unpack MXFP4 to BF16/FP32 for inference.

    Args:
        packed: [out_features, in_features // 2] uint8 packed FP4
        scale_e8m0: [out_features, num_groups] uint8 E8M0 scales
        group_size: MXFP4 group size
        target_dtype: Output dtype

    Returns:
        [out_features, in_features] tensor in target_dtype
    """
    # Step 1: Unpack uint8 -> FP4 codes
    fp4_codes = unpack_uint8_to_fp4(packed)

    # Step 2: Convert FP4 codes to float using lookup table
    table = FP4_E2M1_TABLE.to(fp4_codes.device)
    fp4_codes_long = fp4_codes.to(torch.long)
    values = table[fp4_codes_long]

    # Step 3: Apply E8M0 scales
    in_features = values.shape[-1]
    num_groups = scale_e8m0.shape[-1]

    scales = e8m0_to_float(scale_e8m0).to(values.device)  # [..., num_groups]

    # Reshape values to [..., num_groups, group_size]
    # Trim or pad to match num_groups * group_size
    expected_len = num_groups * group_size
    if in_features > expected_len:
        values = values[..., :expected_len]
    elif in_features < expected_len:
        values = torch.nn.functional.pad(values, (0, expected_len - in_features))

    grouped = values.reshape(*values.shape[:-1], num_groups, group_size)
    grouped = grouped * scales.unsqueeze(-1)

    # Reshape back
    result = grouped.reshape(*values.shape[:-1], num_groups * group_size)

    # Trim to original in_features
    if result.shape[-1] > in_features:
        result = result[..., :in_features]

    return result.to(target_dtype)

#!/usr/bin/env python3
"""
Quanto: Weight Packing Utilities

Unified script for packing, unpacking, and fixing INT4 weights.

Commands:
    pack    - Pack BF16 weights to INT4 format (4x compression)
    fix     - Fix incorrectly packed weights
    unpack  - Unpack INT4 weights back to INT8

Usage:
    python scripts/repack.py pack --input_dir ./quantized_layers --output_dir ./packed_layers
    python scripts/repack.py fix --input_dir ./bad_layers --output_dir ./fixed_layers
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# =============================================================================
# INT4 Packing/Unpacking Functions (Single Source of Truth)
# =============================================================================


def quantize_to_int4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """
    Quantize BF16 weight to INT4 values using per-group quantization.

    Args:
        weight: [out_features, in_features] BF16 tensor
        scale: [out_features, num_groups] scale tensor
        zero_point: [out_features, num_groups] zero point tensor

    Returns:
        [out_features, in_features] INT8 tensor with INT4 values
    """
    out_features, in_features = weight.shape
    num_groups = scale.shape[1]
    group_size = in_features // num_groups

    # Reshape for per-group: [out_features, num_groups, group_size]
    weight_grouped = weight.view(out_features, num_groups, group_size)
    scale_exp = scale.unsqueeze(-1)
    zp_exp = zero_point.unsqueeze(-1).float()

    # Quantize: q = round(w / scale) + zero_point
    quantized = torch.round(weight_grouped.float() / scale_exp) + zp_exp
    quantized = quantized.clamp(-8, 7).to(torch.int8)

    return quantized.view(out_features, in_features)


def pack_int4_to_int32(int4_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack 8 INT4 values into 1 INT32 using Quark/HuggingFace order.

    Args:
        int4_tensor: [out_features, in_features] INT8 tensor with INT4 values

    Returns:
        [out_features, in_features // 8] INT32 packed tensor
    """
    out_features, in_features = int4_tensor.shape

    # Pad in_features to multiple of 8
    if in_features % 8 != 0:
        pad = 8 - (in_features % 8)
        int4_tensor = torch.nn.functional.pad(int4_tensor, (0, pad))

    _, in_f_padded = int4_tensor.shape
    reshaped = int4_tensor.view(out_features, in_f_padded // 8, 8)

    # Pack using Quark order: [0,2,4,6,1,3,5,7]
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros(out_features, in_f_padded // 8, dtype=torch.int32)

    for i, idx in enumerate(order):
        val = reshaped[:, :, idx].to(torch.int32) & 0x0F
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

    # Quark pack order is [0,2,4,6,1,3,5,7]
    order = [0, 2, 4, 6, 1, 3, 5, 7]

    for i in range(8):
        shift = i * 4
        mask = 0x0F << shift
        vals = (packed & mask) >> shift
        # Convert from unsigned to signed INT4
        vals = vals.to(torch.int8)
        vals = torch.where(vals > 7, vals - 16, vals)
        # Put value from position i to index order[i]
        unpacked[:, order[i] :: 8] = vals

    return unpacked


# =============================================================================
# Layer Processing Functions
# =============================================================================


def pack_layer_weights(layer_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Pack all Linear layer weights in a layer to INT4 format.

    Input format (from quantization):
        - weight: [out_features, in_features] BF16
        - weight_quantizer.scale: [out_features, num_groups] BF16
        - weight_quantizer.zero_point: [out_features, num_groups] INT32

    Output format (packed):
        - weight.packed: [out_features, in_features // 8] INT32
        - weight.scale: [out_features, num_groups] BF16
        - weight.zero_point: [out_features, num_groups] INT32
    """
    packed_weights = {}

    # Find Linear weights (2D tensors with quantizer params)
    weight_keys = [
        k
        for k in layer_weights
        if k.endswith(".weight") and "_quantizer" not in k and len(layer_weights[k].shape) == 2
    ]

    for weight_key in weight_keys:
        scale_key = f"{weight_key}_quantizer.scale"
        zp_key = f"{weight_key}_quantizer.zero_point"

        if scale_key not in layer_weights:
            # Not quantized, copy as-is
            packed_weights[weight_key] = layer_weights[weight_key]
            continue

        weight = layer_weights[weight_key]
        scale = layer_weights[scale_key]
        zero_point = layer_weights[zp_key]

        # Quantize to INT4
        int4_vals = quantize_to_int4(weight, scale, zero_point)

        # Pack to INT32
        packed = pack_int4_to_int32(int4_vals)

        # Save with packed format
        packed_weights[f"{weight_key}.packed"] = packed
        packed_weights[f"{weight_key}.scale"] = scale
        packed_weights[f"{weight_key}.zero_point"] = zero_point

    # Copy non-Linear weights (norms, etc.)
    for key, tensor in layer_weights.items():
        if "_quantizer" not in key and key not in weight_keys:
            packed_weights[key] = tensor

    return packed_weights


def fix_layer_weights(layer_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Fix incorrectly packed weights by unpacking, transposing, and repacking.

    The original packed format was [in_features, out_features // 8] (wrong).
    We need to convert to [out_features, in_features // 8] (correct).
    """
    fixed_weights = {}

    # Find packed weight keys
    packed_keys = [k for k in layer_weights if k.endswith(".weight.packed")]

    for packed_key in packed_keys:
        scale_key = packed_key.replace(".weight.packed", ".weight.scale")
        zp_key = packed_key.replace(".weight.packed", ".weight.zero_point")

        if scale_key not in layer_weights:
            fixed_weights[packed_key] = layer_weights[packed_key]
            continue

        packed = layer_weights[packed_key]
        scale = layer_weights[scale_key]

        # Get correct dimensions from scale
        out_features = scale.shape[0]
        num_groups = scale.shape[1]
        in_features = num_groups * 128  # Assuming group_size = 128

        # Unpack: [in_features, out_features // 8] -> [in_features, out_features]
        unpacked = unpack_int32_to_int4(packed)

        # Transpose: [in_features, out_features] -> [out_features, in_features]
        unpacked_t = unpacked.T.contiguous()

        # Re-pack correctly: [out_features, in_features // 8]
        repacked = pack_int4_to_int32(unpacked_t)

        # Save
        weight_key = packed_key[:-7]  # Remove '.packed'
        fixed_weights[f"{weight_key}.packed"] = repacked
        fixed_weights[f"{weight_key}.scale"] = scale
        if zp_key in layer_weights:
            fixed_weights[f"{weight_key}.zero_point"] = layer_weights[zp_key]

    # Copy non-quantized weights
    for key, tensor in layer_weights.items():
        if (
            "_quantizer" not in key
            and not key.endswith(".weight.packed")
            and not key.endswith(".weight.scale")
            and not key.endswith(".weight.zero_point")
        ):
            fixed_weights[key] = tensor

    return fixed_weights


# =============================================================================
# Main Commands
# =============================================================================


def cmd_pack(args: argparse.Namespace) -> dict[str, Any]:
    """Pack BF16 weights to INT4 format."""
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # Determine input structure
    if (input_path / "quantized_layers").exists():
        layer_dir = input_path / "quantized_layers"
        use_subdir = True
    else:
        layer_dir = input_path
        use_subdir = False

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    if use_subdir:
        (output_path / "quantized_layers").mkdir(exist_ok=True)

    # Find layer files
    layer_files = sorted(
        layer_dir.glob("layer_*.safetensors"), key=lambda x: int(x.stem.split("_")[1])
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Found {len(layer_files)} layer files")

    total_original = 0
    total_packed = 0

    for layer_file in tqdm(layer_files, desc="Packing layers"):
        # Load layer
        layer_weights = {}
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                layer_weights[key] = f.get_tensor(key)

        # Calculate original size
        original_size = sum(t.numel() * t.element_size() for t in layer_weights.values())
        total_original += original_size

        # Pack to INT4
        packed_weights = pack_layer_weights(layer_weights)

        # Calculate packed size
        packed_size = sum(t.numel() * t.element_size() for t in packed_weights.values())
        total_packed += packed_size

        # Save
        if use_subdir:
            output_file = output_path / "quantized_layers" / layer_file.name
        else:
            output_file = output_path / layer_file.name
        save_file(packed_weights, str(output_file))

    # Copy metadata
    result_file = input_path / "quantization_result.json"
    if not result_file.exists():
        result_file = input_path.parent / "quantization_result.json"

    result_data = {}
    if result_file.exists():
        result_data = json.loads(result_file.read_text())
        shutil.copy(result_file, output_path / "quantization_result.json")

    # Copy config and tokenizer from original model
    if "model_path" in result_data:
        model_path = Path(result_data["model_path"])
        for fname in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
        ]:
            src = model_path / fname
            if src.exists():
                shutil.copy(src, output_path / fname)

    # Update result with packed info
    compression = total_original / total_packed if total_packed > 0 else 0
    result_data["packed"] = True
    result_data["original_size_gb"] = total_original / 1024**3
    result_data["packed_size_gb"] = total_packed / 1024**3
    result_data["compression_ratio"] = compression

    with open(output_path / "quantization_result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PACKING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Original size: {total_original / 1024**3:.2f} GB")
    print(f"Packed size: {total_packed / 1024**3:.2f} GB")
    print(f"Compression: {compression:.2f}x")
    print(f"Output: {output_path}")

    return result_data


def cmd_fix(args: argparse.Namespace) -> dict[str, Any]:
    """Fix incorrectly packed weights."""
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    if (input_path / "quantized_layers").exists():
        layer_dir = input_path / "quantized_layers"
        (output_path / "quantized_layers").mkdir(exist_ok=True)
        use_subdir = True
    else:
        layer_dir = input_path
        use_subdir = False

    # Find layer files
    layer_files = sorted(
        layer_dir.glob("layer_*.safetensors"), key=lambda x: int(x.stem.split("_")[1])
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Found {len(layer_files)} layer files")

    for layer_file in tqdm(layer_files, desc="Fixing layers"):
        # Load layer
        layer_weights = {}
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                layer_weights[key] = f.get_tensor(key)

        # Fix packed weights
        fixed_weights = fix_layer_weights(layer_weights)

        # Save
        if use_subdir:
            output_file = output_path / "quantized_layers" / layer_file.name
        else:
            output_file = output_path / layer_file.name
        save_file(fixed_weights, str(output_file))

    # Copy metadata
    for fname in ["quantization_result.json"]:
        src = input_path / fname
        if src.exists():
            shutil.copy(src, output_path / fname)

    print(f"\nDone! Fixed layers saved to {output_path}")

    return {"success": True, "output_dir": str(output_path)}


def cmd_unpack(args: argparse.Namespace) -> dict[str, Any]:
    """Unpack INT4 weights back to INT8 (for debugging/verification)."""
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Find layer files
    layer_files = sorted(
        input_path.glob("**/layer_*.safetensors"), key=lambda x: int(x.stem.split("_")[1])
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Found {len(layer_files)} layer files")

    for layer_file in tqdm(layer_files, desc="Unpacking layers"):
        # Load layer
        layer_weights = {}
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                layer_weights[key] = f.get_tensor(key)

        # Unpack INT4 weights
        unpacked_weights = {}
        for key, tensor in layer_weights.items():
            if key.endswith(".weight.packed"):
                # Unpack
                unpacked = unpack_int32_to_int4(tensor)
                weight_key = key[:-7]  # Remove '.packed'
                unpacked_weights[weight_key] = unpacked
            else:
                unpacked_weights[key] = tensor

        # Save
        output_file = output_path / layer_file.name
        save_file(unpacked_weights, str(output_file))

    print(f"\nDone! Unpacked layers saved to {output_path}")

    return {"success": True, "output_dir": str(output_path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quanto Weight Packing Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Pack command
    pack_parser = subparsers.add_parser("pack", help="Pack BF16 weights to INT4 format")
    pack_parser.add_argument(
        "--input_dir", required=True, help="Input directory with quantized layers"
    )
    pack_parser.add_argument(
        "--output_dir", required=True, help="Output directory for packed layers"
    )

    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix incorrectly packed weights")
    fix_parser.add_argument(
        "--input_dir", required=True, help="Input directory with incorrect layers"
    )
    fix_parser.add_argument("--output_dir", required=True, help="Output directory for fixed layers")

    # Unpack command
    unpack_parser = subparsers.add_parser("unpack", help="Unpack INT4 weights back to INT8")
    unpack_parser.add_argument(
        "--input_dir", required=True, help="Input directory with packed layers"
    )
    unpack_parser.add_argument(
        "--output_dir", required=True, help="Output directory for unpacked layers"
    )

    args = parser.parse_args()

    if args.command == "pack":
        result = cmd_pack(args)
        return 0 if result.get("packed") else 1
    elif args.command == "fix":
        result = cmd_fix(args)
        return 0 if result.get("success") else 1
    elif args.command == "unpack":
        result = cmd_unpack(args)
        return 0 if result.get("success") else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())

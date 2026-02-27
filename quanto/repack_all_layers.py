#!/usr/bin/env python3
"""
Repack All Quantized Layers to INT4

Converts BF16 weights + scale/zero_point to packed INT4 format.
Achieves ~4x compression.

Usage:
    python repack_all_layers.py --input_dir /output/qwen3-32b-lazy-int4-cuda-b8 --output_dir /output/qwen3-32b-int4-packed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def quantize_to_int4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Quantize BF16 weight to INT4 values."""
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
    """Pack 8 INT4 values into 1 INT32 (Quark order)."""
    out_features, in_features = int4_tensor.shape

    # Transpose for Quark format
    int4_tensor = int4_tensor.T.contiguous()

    # Pad to multiple of 8
    if out_features % 8 != 0:
        pad = 8 - (out_features % 8)
        int4_tensor = torch.nn.functional.pad(int4_tensor, (0, pad))

    in_f, out_f = int4_tensor.shape
    reshaped = int4_tensor.view(in_f, out_f // 8, 8)

    # Pack using Quark order: [0,2,4,6,1,3,5,7]
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros(in_f, out_f // 8, dtype=torch.int32)

    for i, idx in enumerate(order):
        val = (reshaped[:, :, idx].to(torch.int32) & 0x0F)
        packed = packed | (val << (i * 4))

    return packed


def pack_layer_weights(layer_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pack all Linear layer weights in a layer to INT4."""
    packed_weights = {}

    # Find Linear weights (2D tensors with quantizer params)
    weight_keys = [k for k in layer_weights.keys()
                   if k.endswith('.weight') and '_quantizer' not in k and len(layer_weights[k].shape) == 2]

    for weight_key in weight_keys:
        scale_key = f'{weight_key}_quantizer.scale'
        zp_key = f'{weight_key}_quantizer.zero_point'

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
        packed_weights[f'{weight_key}.packed'] = packed
        packed_weights[f'{weight_key}.scale'] = scale
        packed_weights[f'{weight_key}.zero_point'] = zero_point

    # Copy non-Linear weights (norms, etc.)
    for key, tensor in layer_weights.items():
        if '_quantizer' not in key and key not in weight_keys:
            packed_weights[key] = tensor

    return packed_weights


def repack_all_layers(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """Repack all quantized layers to INT4 format."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "quantized_layers").mkdir(exist_ok=True)

    # Find all layer files
    layer_files = sorted(
        input_path.glob("quantized_layers/layer_*.safetensors"),
        key=lambda x: int(x.stem.split("_")[1])
    )

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
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
        output_file = output_path / "quantized_layers" / layer_file.name
        save_file(packed_weights, str(output_file))

    # Copy metadata
    result_file = input_path / "quantization_result.json"
    if result_file.exists():
        import shutil
        shutil.copy(result_file, output_path / "quantization_result.json")

    # Copy config and tokenizer from original model path
    result_data = {}
    if result_file.exists():
        with open(result_file) as f:
            result_data = json.load(f)

    if "model_path" in result_data:
        model_path = Path(result_data["model_path"])
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "generation_config.json", "tokenizer.model", "special_tokens_map.json"]:
            src = model_path / fname
            if src.exists():
                import shutil
                shutil.copy(src, output_path / fname)

    # Update result with packed info
    result_data["packed"] = True
    result_data["original_size_mb"] = total_original / 1024**2
    result_data["packed_size_mb"] = total_packed / 1024**2
    result_data["compression_ratio"] = total_original / total_packed if total_packed > 0 else 0

    with open(output_path / "quantization_result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n{'='*60}")
    print("REPACKING COMPLETE")
    print(f"{'='*60}")
    print(f"Original size: {total_original / 1024**3:.2f} GB")
    print(f"Packed size: {total_packed / 1024**3:.2f} GB")
    print(f"Compression: {total_original / total_packed:.2f}x")
    print(f"Output: {output_dir}")

    return result_data


def main():
    parser = argparse.ArgumentParser(description="Repack quantized layers to INT4")
    parser.add_argument("--input_dir", required=True, help="Input directory with quantized layers")
    parser.add_argument("--output_dir", required=True, help="Output directory for packed layers")

    args = parser.parse_args()

    return repack_all_layers(args.input_dir, args.output_dir)


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("packed") else 1)

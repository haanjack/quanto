#!/usr/bin/env python3
"""
Repack Quantized Layers to INT4

This script takes the existing quantized layer files (with BF16 weights + scale/zero_point)
and repacks them into actual INT4 format for 4x size reduction.

Output:
- .weight.packed: INT32 tensor with packed INT4 values
- .weight.scale: Scale tensor (BF16)
- .weight.zero_point: Zero point tensor (INT32)
"""

import sys
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from int4_pack import pack_layer_weights


def repack_quantized_layers(
    input_dir: str,
    output_dir: str,
    group_size: int = 128,
) -> dict:
    """
    Repack quantized layers from BF16 to packed INT4 format.

    Args:
        input_dir: Directory with original quantized layers
        output_dir: Directory for repacked INT4 layers
        group_size: Quantization group size (default 128)

    Returns:
        Summary dict
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find layer files
    layer_files = sorted(
        input_path.glob("layer_*.safetensors"),
        key=lambda x: int(x.stem.split("_")[1])
    )

    print(f"Found {len(layer_files)} layer files")
    print(f"Group size: {group_size}")

    total_original = 0
    total_packed = 0

    for layer_file in tqdm(layer_files, desc="Repacking layers"):
        # Load layer weights
        layer_weights = {}
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                layer_weights[key] = f.get_tensor(key)

        # Calculate original size
        original_size = sum(t.numel() * t.element_size() for t in layer_weights.values())
        total_original += original_size

        # Pack to INT4
        packed_weights = pack_layer_weights(layer_weights, group_size)

        # Calculate packed size
        packed_size = sum(t.numel() * t.element_size() for t in packed_weights.values())
        total_packed += packed_size

        # Save packed layer
        output_file = output_path / layer_file.name
        save_file(packed_weights, str(output_file))

    # Copy quantization result
    result_file = input_path.parent / "quantization_result.json"
    if result_file.exists():
        import shutil
        shutil.copy(result_file, output_path.parent / "quantization_result.json")

    # Calculate compression
    compression = total_original / total_packed if total_packed > 0 else 0

    print(f"\nRepacking complete!")
    print(f"  Original size: {total_original / 1024**3:.2f} GB")
    print(f"  Packed size: {total_packed / 1024**3:.2f} GB")
    print(f"  Compression: {compression:.2f}x")
    print(f"  Output: {output_dir}")

    return {
        "success": True,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_layers": len(layer_files),
        "original_size_gb": total_original / 1024**3,
        "packed_size_gb": total_packed / 1024**3,
        "compression_ratio": compression,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Repack quantized layers to INT4")
    parser.add_argument("--input_dir", required=True, help="Input directory with quantized layers")
    parser.add_argument("--output_dir", required=True, help="Output directory for INT4 layers")
    parser.add_argument("--group_size", type=int, default=128, help="Quantization group size")

    args = parser.parse_args()

    result = repack_quantized_layers(args.input_dir, args.output_dir, args.group_size)

    # Save result
    with open(Path(args.output_dir).parent / "repack_result.json", "w") as f:
        json.dump(result, f, indent=2)

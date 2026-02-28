#!/usr/bin/env python3
"""Test packing a single layer to INT4."""

from __future__ import annotations

import argparse

from safetensors import safe_open
from safetensors.torch import save_file

from quanto.utils.int4_pack import pack_int4_to_int32, quantize_to_int4


def test_pack_layer(layer_file: str, output_file: str = "/tmp/layer_0_int4.safetensors") -> None:
    """Test packing a single layer to INT4.

    Args:
        layer_file: Path to the layer safetensors file
        output_file: Path to save the packed layer
    """
    # Load layer
    layer_weights = {}
    with safe_open(layer_file, framework="pt", device="cpu") as f:
        for key in f:
            layer_weights[key] = f.get_tensor(key)

    # Find Linear weights
    weight_keys = [
        k
        for k in layer_weights
        if k.endswith(".weight") and "_quantizer" not in k and len(layer_weights[k].shape) == 2
    ]

    packed_weights = {}

    for weight_key in weight_keys:
        scale_key = f"{weight_key}_quantizer.scale"
        zp_key = f"{weight_key}_quantizer.zero_point"

        weight = layer_weights[weight_key]
        scale = layer_weights[scale_key]
        zero_point = layer_weights[zp_key]

        # Quantize to INT4
        int4_vals = quantize_to_int4(weight, scale, zero_point)

        # Pack to INT32
        packed = pack_int4_to_int32(int4_vals)

        # Save with clean names (matching HuggingFace format)
        packed_weights[f"{weight_key}.packed"] = packed
        packed_weights[f"{weight_key}.scale"] = scale
        packed_weights[f"{weight_key}.zero_point"] = zero_point

    # Copy non-Linear weights (norms)
    for key, tensor in layer_weights.items():
        if "_quantizer" not in key and key not in weight_keys:
            packed_weights[key] = tensor

    # Save
    save_file(packed_weights, output_file)

    # Verify
    print(f"Packed layer saved to {output_file}")
    print("\nContents:")
    for k, v in sorted(packed_weights.items()):
        print(f"  {k}: {list(v.shape)}, {v.dtype}")

    # Size
    size = sum(v.numel() * v.element_size() for v in packed_weights.values())
    print(f"\nTotal size: {size / 1024**2:.2f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test packing a single layer to INT4")
    parser.add_argument(
        "--layer_file",
        type=str,
        default="output/qwen3-32b-lazy-int4-cuda-b8/quantized_layers/layer_0.safetensors",
        help="Path to layer safetensors file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/layer_0_int4.safetensors",
        help="Output path for packed layer",
    )

    args = parser.parse_args()
    test_pack_layer(args.layer_file, args.output)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

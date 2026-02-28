#!/usr/bin/env python3
"""
Test script for layer-wise quantization with Qwen3-32B using 1 GPU.
This uses the pipeline quantization approach that loads one layer at a time.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from quanto.core.layerwise_quant import LayerwiseQuantizer


def main() -> int:
    # Configuration for Qwen3-32B with layer-wise quantization
    config = {
        "model_path": "/models/qwen3-32b",
        "output_dir": "/output/qwen3-32b-layerwise-int4",
        "precision": "int4",
        "calibration_data": "pileval",
        "num_calib_samples": 32,  # Reduced for memory efficiency
        "seq_len": 512,
        "batch_size": 1,
        "device": "cuda:0",  # Use single GPU
        "exclude_layers": ["lm_head"],  # Start with minimal exclusions
        "trust_remote_code": True,
    }

    print("=" * 60)
    print("Layer-wise Quantization Test for Qwen3-32B")
    print("=" * 60)
    print(f"Model: {config['model_path']}")
    print(f"Output: {config['output_dir']}")
    print(f"Precision: {config['precision']}")
    print(f"Device: {config['device']}")
    print(f"Calibration samples: {config['num_calib_samples']}")
    print("=" * 60)

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Create layerwise quantizer
    quantizer = LayerwiseQuantizer(**config)

    # Run pipeline quantization (GPU-only computation)
    result = quantizer.run_pipeline_quantization()

    # Save result
    result_path = Path(config["output_dir"]) / "quantization_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved to: {result_path}")

    if result.get("success", False):
        print("\n" + "=" * 60)
        print("QUANTIZATION SUCCESSFUL")
        print("=" * 60)
        print(f"Model type: {result.get('model_type', 'unknown')}")
        print(f"Quant scheme: {result.get('quant_scheme', 'unknown')}")
        print(f"Total time: {result.get('timing', {}).get('total', 0):.2f}s")
        print(f"Output directory: {config['output_dir']}")
        print("=" * 60)
        return 0
    else:
        print("\nQuantization FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

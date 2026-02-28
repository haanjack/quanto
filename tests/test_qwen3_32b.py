#!/usr/bin/env python3
"""
Test script for iterative quantization with Qwen3-32B.
This script runs inside the Docker container.

Note: For 32B models, we use memory-efficient loading and reduced calibration samples.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from quanto.core.iterative_quantizer import IterativeConfig, IterativeQuantizer


def main() -> int:
    # Configuration for Qwen3-32B
    config = IterativeConfig(
        model_path="/models/qwen3-32b",
        output_dir="/output/qwen3-32b-iterative-int4",
        precision="int4",
        calibration_data="pileval",
        num_calib_samples=32,  # Reduced for memory efficiency
        seq_len=512,
        batch_size=1,
        device="cuda",
        memory_efficient=True,  # Use device_map="auto" for large models
        max_iterations=3,
        target_ppl_degradation=1.5,  # Slightly higher tolerance for larger model
        max_exclude_percent=15.0,
        initial_exclusions=["lm_head"],  # Start with minimal exclusions
        skip_evaluation=False,
        debug_dir="/output/debug_qwen3",
    )

    print("=" * 60)
    print("Iterative Quantization Test for Qwen3-32B")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Output: {config.output_dir}")
    print(f"Precision: {config.precision}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Target PPL degradation: {config.target_ppl_degradation}")
    print(f"Memory efficient: {config.memory_efficient}")
    print(f"Calibration samples: {config.num_calib_samples}")
    print("=" * 60)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.debug_dir, exist_ok=True)

    # Run iterative quantization
    quantizer = IterativeQuantizer(config)
    result = quantizer.run()

    # Save result
    result_path = Path(config.output_dir) / "iterative_result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResult saved to: {result_path}")

    # Print final exclusion list
    if result.success:
        exclusions_path = Path(config.output_dir) / "exclusion_list.json"
        exclusion_data = {
            "model_type": result.model_type,
            "final_exclude_layers": result.final_exclude_layers,
            "best_ppl_degradation": result.best_ppl_degradation,
            "best_iteration": result.best_iteration,
        }
        with open(exclusions_path, "w") as f:
            json.dump(exclusion_data, f, indent=2)
        print(f"Exclusion list saved to: {exclusions_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

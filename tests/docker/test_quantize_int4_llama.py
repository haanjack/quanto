#!/usr/bin/env python3
"""
Test: Quantize Llama3.1-8B-Instruct to INT4

Model: ~/models/meta-llama/Llama3.1-8b-instruct
Output: ./outputs/llama3.1-8b-int4
Evaluation: GSM-8K (5-shot, strict-match) with lm-eval
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from quanto.core.auto_quantize import AutoQuantizer, QuantizationConfig


def run_gsm8k_evaluation(model_path: str, output_file: str) -> dict:
    """Run GSM-8K evaluation using lm-eval."""
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", "gsm8k",
        "--num_fewshot", "5",
        "--batch_size", "auto",
        "--output_path", output_file,
    ]

    print(f"Running GSM-8K evaluation: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse results
    results_file = Path(output_file) / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    return {"error": result.stderr, "stdout": result.stdout}


def main() -> int:
    print("=" * 60)
    print("Test: Quantize Meta-Llama-3-8B to INT4")
    print("=" * 60)

    model_path = Path("/models/meta-llama/Meta-Llama-3-8B")
    output_dir = Path("/output/llama3-8b-int4")

    # Verify model exists
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = QuantizationConfig(
        model_path=str(model_path),
        output_dir=str(output_dir),
        precision="int4",
        calibration_data="pileval",
        num_calib_samples=128,
        seq_len=512,
        batch_size=1,
        device="cuda",
        exclude_layers=["lm_head"],
        skip_evaluation=True,  # We'll use GSM-8K instead
        trust_remote_code=True,
    )

    # Run quantization
    print("\n[1/3] Running quantization...")
    start_time = time.time()

    quantizer = AutoQuantizer(config)
    result = quantizer.run()

    quant_time = time.time() - start_time
    print(f"Quantization completed in {quant_time:.2f}s")

    if not result.success:
        print(f"ERROR: Quantization failed: {result.error_message}")
        # Clean up failed output
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return 1

    # Save quantization result
    with open(output_dir / "quantization_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Quantized model saved to: {output_dir}")

    # Run GSM-8K evaluation
    print("\n[2/3] Running GSM-8K evaluation...")
    eval_start = time.time()

    eval_results = run_gsm8k_evaluation(str(output_dir), str(output_dir / "eval"))

    eval_time = time.time() - eval_start
    print(f"Evaluation completed in {eval_time:.2f}s")

    # Extract GSM-8K score
    gsm8k_score = None
    if "results" in eval_results:
        gsm8k_result = eval_results["results"].get("gsm8k", {})
        gsm8k_score = gsm8k_result.get("exact_match,strict_match", gsm8k_result.get("acc"))

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: Meta-Llama-3-8B")
    print(f"Precision: INT4")
    print(f"Quantization time: {quant_time:.2f}s")
    print(f"Output directory: {output_dir}")
    if gsm8k_score is not None:
        print(f"GSM-8K (5-shot, strict-match): {gsm8k_score:.4f}")
    else:
        print("GSM-8K: Evaluation failed or not available")
    print("=" * 60)

    # Save test result
    test_result = {
        "test": "quantize_int4_llama",
        "model": "Meta-Llama-3-8B",
        "precision": "int4",
        "success": result.success,
        "quantization_time": quant_time,
        "evaluation_time": eval_time,
        "gsm8k_score": gsm8k_score,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "test_result.json", "w") as f:
        json.dump(test_result, f, indent=2)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

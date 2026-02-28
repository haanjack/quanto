#!/usr/bin/env python3
"""
Test: Dequantize Llama3.1-8B-Instruct INT4 to BF16

Input: ./outputs/llama3.1-8b-int4
Output: ./outputs/llama3.1-8b-bf16
Evaluation: GSM-8K (5-shot, strict-match) with lm-eval
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from quanto.core.dequantize import ModelDequantizer, DequantizationConfig


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

    result = subprocess.run(cmd, capture_output=True, text=True)

    results_file = Path(output_file) / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    return {"error": result.stderr, "stdout": result.stdout}


def main() -> int:
    print("=" * 60)
    print("Test: Dequantize Meta-Llama-3-8B INT4 to BF16")
    print("=" * 60)

    input_dir = Path("/output/llama3-8b-int4")
    output_dir = Path("/output/llama3-8b-bf16")

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        print("Run test_quantize_int4_llama.py first")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dequantization config
    config = DequantizationConfig(
        model_path=str(input_dir),
        output_dir=str(output_dir),
        output_dtype="bf16",
        device="cuda",
    )

    # Run dequantization
    print("\n[1/3] Running dequantization...")
    start_time = time.time()

    dequantizer = ModelDequantizer(config)
    result = dequantizer.dequantize()

    dequant_time = time.time() - start_time
    print(f"Dequantization completed in {dequant_time:.2f}s")

    if not result.success:
        print(f"ERROR: Dequantization failed: {result.error_message}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return 1

    # Run GSM-8K evaluation
    print("\n[2/3] Running GSM-8K evaluation...")
    eval_start = time.time()

    eval_results = run_gsm8k_evaluation(str(output_dir), str(output_dir / "eval"))

    eval_time = time.time() - eval_start
    print(f"Evaluation completed in {eval_time:.2f}s")

    gsm8k_score = None
    if "results" in eval_results:
        gsm8k_result = eval_results["results"].get("gsm8k", {})
        gsm8k_score = gsm8k_result.get("exact_match,strict_match", gsm8k_result.get("acc"))

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: Meta-Llama-3-8B")
    print(f"Operation: INT4 -> BF16 (dequantization)")
    print(f"Dequantization time: {dequant_time:.2f}s")
    print(f"Output directory: {output_dir}")
    if gsm8k_score is not None:
        print(f"GSM-8K (5-shot, strict-match): {gsm8k_score:.4f}")
    else:
        print("GSM-8K: Evaluation failed or not available")
    print("=" * 60)

    test_result = {
        "test": "dequantize_llama",
        "model": "Meta-Llama-3-8B",
        "operation": "int4_to_bf16",
        "success": result.success,
        "dequantization_time": dequant_time,
        "evaluation_time": eval_time,
        "gsm8k_score": gsm8k_score,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "test_result.json", "w") as f:
        json.dump(test_result, f, indent=2)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

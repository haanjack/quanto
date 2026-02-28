#!/usr/bin/env python3
"""
Test: Verify HuggingFace Transformers Compatibility

This test verifies that quantized models can be loaded and run inference
using HuggingFace Transformers without any quantization-specific code.

Tests:
1. Load quantized model with AutoModelForCausalLM
2. Run simple inference (text generation)
3. Verify output is valid
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def verify_hf_compatibility(model_path: str) -> dict:
    """Verify model can be loaded and used with HuggingFace Transformers."""
    results = {
        "load_success": False,
        "inference_success": False,
        "output_valid": False,
        "error": None,
        "load_time": 0,
        "inference_time": 0,
    }

    try:
        # Test 1: Load model
        print(f"Loading model from {model_path}...")
        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        results["load_time"] = time.time() - start_time
        results["load_success"] = True
        print(f"Model loaded in {results['load_time']:.2f}s")

        # Test 2: Run inference
        print("Running inference test...")
        test_prompt = "The capital of France is"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        results["inference_time"] = time.time() - start_time
        results["inference_success"] = True

        # Test 3: Verify output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {test_prompt}")
        print(f"Output: {generated_text}")

        # Basic validation - output should be different from input
        results["output_valid"] = len(generated_text) > len(test_prompt)
        results["generated_text"] = generated_text

        print(f"Inference completed in {results['inference_time']:.2f}s")

    except Exception as e:
        results["error"] = str(e)
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    return results


def main() -> int:
    print("=" * 60)
    print("Test: Verify HuggingFace Transformers Compatibility")
    print("=" * 60)

    test_models = [
        ("Llama3.1-8B-INT4", "/output/llama3.1-8b-int4"),
        ("Llama3.1-8B-INT8", "/output/llama3.1-8b-int8"),
        ("Qwen3-32B-INT4", "/output/qwen3-32b-int4"),
        ("Llama3.1-8B-BF16 (dequantized)", "/output/llama3.1-8b-bf16"),
    ]

    all_results = {}

    for model_name, model_path in test_models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"Path: {model_path}")
        print("=" * 60)

        if not Path(model_path).exists():
            print(f"SKIPPED: Model not found at {model_path}")
            all_results[model_name] = {"skipped": True, "reason": "Model not found"}
            continue

        result = verify_hf_compatibility(model_path)
        all_results[model_name] = result

        status = "✓ PASSED" if result["load_success"] and result["inference_success"] and result["output_valid"] else "✗ FAILED"
        print(f"\nStatus: {status}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model_name, result in all_results.items():
        if result.get("skipped"):
            print(f"  {model_name}: SKIPPED ({result.get('reason', 'unknown')})")
        elif result.get("load_success") and result.get("inference_success") and result.get("output_valid"):
            print(f"  {model_name}: ✓ PASSED")
        else:
            print(f"  {model_name}: ✗ FAILED ({result.get('error', 'unknown error')})")

    # Save results
    output_file = Path("/output/hf_verify_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Return success only if all non-skipped tests passed
    failed = any(
        not r.get("skipped") and not (r.get("load_success") and r.get("inference_success") and r.get("output_valid"))
        for r in all_results.values()
    )

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

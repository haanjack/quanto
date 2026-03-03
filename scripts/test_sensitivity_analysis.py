#!/usr/bin/env python3
"""
End-to-End Test for Sequential Sensitivity Analysis.

This script tests the sensitivity analysis feature:
1. Runs sensitivity analysis to identify sensitive layers
2. Quantizes with sensitivity-based layer exclusion
3. Evaluates quantized models with MMLU 5-shot
4. Reports activation cache performance

Usage:
    python scripts/test_sensitivity_analysis.py --model llama3 --device cuda
    python scripts/test_sensitivity_analysis.py --model qwen3 --device cuda --skip-quantize
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_model_path(model_name: str) -> str:
    """Get model path based on model name."""
    home = os.path.expanduser("~")
    models = {
        "llama3": f"{home}/models/meta-llama/Meta-Llama-3-8B",
        "qwen3": f"{home}/models/qwen/qwen3-32b",
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    path = models[model_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return path


def run_sensitivity_analysis(config) -> dict:
    """
    Run sensitivity analysis and return results.
    """
    from quanto.core.sensitivity import SequentialSensitivityAnalyzer

    print("\n" + "=" * 60)
    print("PHASE 1: Sensitivity Analysis")
    print("=" * 60)

    analyzer = SequentialSensitivityAnalyzer(
        config=config,
        cache_on_gpu=config.sensitivity_cache_on_gpu,
    )

    start_time = time.time()
    result = analyzer.analyze()
    elapsed = time.time() - start_time

    # Get cache stats
    cache_stats = {
        "num_entries": analyzer.cache.stats.num_entries,
        "gpu_memory_gb": analyzer.cache.stats.gpu_memory_gb,
        "cpu_memory_gb": analyzer.cache.stats.cpu_memory_gb,
        "hits": analyzer.cache.stats.hits,
        "misses": analyzer.cache.stats.misses,
    }

    print(f"\nSensitivity Analysis Results:")
    print(f"  Success: {result.success}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Layers analyzed: {len(result.sensitive_layers)}")
    print(f"  Layers to exclude: {len(result.excluded_layers)}")
    print(f"\nCache Performance:")
    print(f"  {analyzer.cache.get_memory_summary()}")
    hit_rate = cache_stats['hits'] / max(cache_stats['hits'] + cache_stats['misses'], 1) * 100
    print(f"  Hit rate: {hit_rate:.1f}%")

    if result.excluded_layers:
        print(f"\nExcluded layers ({len(result.excluded_layers)}):")
        for layer in result.excluded_layers[:10]:
            score = result.sensitive_layers.get(layer, 0)
            print(f"  {layer}: {score:.6f}")
        if len(result.excluded_layers) > 10:
            print(f"  ... and {len(result.excluded_layers) - 10} more")

    return {
        "success": result.success,
        "sensitive_layers": result.sensitive_layers,
        "excluded_layers": result.excluded_layers,
        "cache_stats": cache_stats,
        "timing": result.timing,
        "elapsed": elapsed,
    }


def run_quantization(config, sensitivity_result: dict) -> dict:
    """
    Run quantization with sensitivity-based exclusion.
    """
    from quanto import UnifiedQuantizer

    print("\n" + "=" * 60)
    print("PHASE 2: Quantization with Sensitivity Exclusion")
    print("=" * 60)

    # Update config to use discovered exclusions
    if sensitivity_result["excluded_layers"]:
        # Merge with default exclusions
        default_exclude = ["lm_head", "*embed*", "*norm*"]
        all_exclude = list(set(default_exclude + sensitivity_result["excluded_layers"]))
        config.exclude_layers = all_exclude
        print(f"Excluding {len(all_exclude)} layers total")

    # Disable sensitivity analysis since we already have results
    config.sensitivity_analysis = False

    quantizer = UnifiedQuantizer(config)

    start_time = time.time()
    result = quantizer.run()
    elapsed = time.time() - start_time

    print(f"\nQuantization Results:")
    print(f"  Success: {result.success}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output: {result.output_dir}")
    if result.exclude_layers_used:
        print(f"  Exclude layers used: {len(result.exclude_layers_used)}")

    return {
        "success": result.success,
        "output_dir": result.output_dir,
        "exclude_layers_used": result.exclude_layers_used,
        "timing": result.timing,
        "elapsed": elapsed,
    }


def run_evaluation(output_dir: str) -> dict:
    """
    Run MMLU 5-shot evaluation on quantized model.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: MMLU 5-shot Evaluation")
    print("=" * 60)

    try:
        start_time = time.time()

        # Run lm_eval
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={output_dir},dtype=float16,trust_remote_code=True",
            "--tasks", "mmlu",
            "--num_fewshot", "5",
            "--batch_size", "4",
            "--output_path", f"{output_dir}/mmlu_results",
        ]

        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        # Parse results
        accuracy = None
        if result.returncode == 0:
            # Try to extract accuracy from output
            import re
            match = re.search(r"acc[,|:]\s*([0-9.]+)", result.stdout + result.stderr)
            if match:
                accuracy = float(match.group(1))
            print(f"\nEvaluation completed in {elapsed:.2f}s")
            if accuracy:
                print(f"MMLU 5-shot accuracy: {accuracy:.4f}")
        else:
            print(f"Evaluation failed with code {result.returncode}")
            print(f"Error: {result.stderr[:500]}")

        return {
            "success": result.returncode == 0,
            "accuracy": accuracy,
            "elapsed": elapsed,
            "stdout": result.stdout[:2000] if result.stdout else None,
            "stderr": result.stderr[:500] if result.stderr else None,
        }

    except FileNotFoundError:
        print("lm_eval not found. Skipping evaluation.")
        return {"success": False, "error": "lm_eval not installed"}
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {"success": False, "error": str(e)}


def run_baseline_comparison(config) -> dict:
    """
    Run comparison between with and without sensitivity analysis.
    """
    from quanto import UnifiedQuantizer

    print("\n" + "=" * 60)
    print("PHASE 4: Baseline Comparison (without sensitivity)")
    print("=" * 60)

    # Create a copy of config without sensitivity
    config_dict = config.to_dict()
    config_dict["sensitivity_analysis"] = False
    config_dict["sensitivity_threshold"] = 0.0
    config_dict["output_dir"] = f"{config.output_dir}_baseline"

    from quanto import UnifiedConfig
    baseline_config = UnifiedConfig(**config_dict)

    quantizer = UnifiedQuantizer(baseline_config)

    start_time = time.time()
    result = quantizer.run()
    elapsed = time.time() - start_time

    print(f"\nBaseline Quantization Results:")
    print(f"  Success: {result.success}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output: {result.output_dir}")

    return {
        "success": result.success,
        "output_dir": result.output_dir,
        "exclude_layers_used": result.exclude_layers_used,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Sensitivity Analysis")
    parser.add_argument("--model", type=str, required=True,
                        choices=["llama3", "qwen3"],
                        help="Model to test")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Sensitivity threshold")
    parser.add_argument("--calib-samples", type=int, default=128,
                        help="Calibration samples")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip quantization, only run analysis")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip MMLU evaluation")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare with baseline (no sensitivity)")
    args = parser.parse_args()

    # Get model path
    try:
        model_path = get_model_path(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./test_outputs/{args.model}_sensitivity_int4"

    print("=" * 60)
    print(f"End-to-End Sensitivity Analysis Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Path: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    # Import after path is set
    from quanto import UnifiedConfig

    # Create config
    config = UnifiedConfig(
        model_path=model_path,
        output_dir=output_dir,
        precision="int4",
        memory_strategy="auto",
        pack_int4=True,
        sensitivity_analysis=True,
        sensitivity_threshold=args.threshold,
        sensitivity_cache_on_gpu=True,
        num_calib_samples=args.calib_samples,
        skip_evaluation=True,
    )

    # Results container
    results = {
        "model": args.model,
        "config": config.to_dict(),
    }

    # Phase 1: Sensitivity Analysis
    sensitivity_result = run_sensitivity_analysis(config)
    results["sensitivity_analysis"] = sensitivity_result

    if not sensitivity_result["success"]:
        print("\nSensitivity analysis failed. Stopping.")
        print(json.dumps(results, indent=2, default=str))
        sys.exit(1)

    if args.skip_quantize:
        print("\nSkipping quantization (--skip-quantize)")
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(json.dumps(results, indent=2, default=str))
        return

    # Phase 2: Quantization
    quant_result = run_quantization(config, sensitivity_result)
    results["quantization"] = quant_result

    if not quant_result["success"]:
        print("\nQuantization failed. Stopping.")
        print(json.dumps(results, indent=2, default=str))
        sys.exit(1)

    # Phase 3: Evaluation
    if not args.skip_eval:
        eval_result = run_evaluation(quant_result["output_dir"])
        results["evaluation"] = eval_result

    # Phase 4: Baseline comparison (optional)
    if args.compare_baseline:
        baseline_result = run_baseline_comparison(config)
        results["baseline"] = baseline_result

        if not args.skip_eval and baseline_result["success"]:
            baseline_eval = run_evaluation(baseline_result["output_dir"])
            results["baseline_evaluation"] = baseline_eval

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")

    print(f"\nSensitivity Analysis:")
    print(f"  Layers analyzed: {len(sensitivity_result['sensitive_layers'])}")
    print(f"  Layers excluded: {len(sensitivity_result['excluded_layers'])}")
    print(f"  Cache GPU memory: {sensitivity_result['cache_stats']['gpu_memory_gb']:.2f} GB")
    hits = sensitivity_result['cache_stats']['hits']
    misses = sensitivity_result['cache_stats']['misses']
    hit_rate = hits / max(hits + misses, 1) * 100
    print(f"  Cache hit rate: {hit_rate:.1f}%")

    print(f"\nQuantization:")
    print(f"  Output: {quant_result['output_dir']}")
    print(f"  Time: {quant_result['elapsed']:.2f}s")

    if "evaluation" in results and results["evaluation"].get("accuracy"):
        print(f"\nEvaluation:")
        print(f"  MMLU 5-shot: {results['evaluation']['accuracy']:.4f}")

    if "baseline_evaluation" in results and results["baseline_evaluation"].get("accuracy"):
        print(f"\nBaseline (no sensitivity):")
        print(f"  MMLU 5-shot: {results['baseline_evaluation']['accuracy']:.4f}")

    # Save results
    results_file = f"{output_dir}/sensitivity_test_results.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

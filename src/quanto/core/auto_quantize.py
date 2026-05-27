"""
Auto-Quantize: Backward compatibility wrapper for UnifiedQuantizer.

This module provides backward compatibility for code using AutoQuantizer.
New code should use UnifiedQuantizer and UnifiedConfig directly.

Migration Guide:
    # Old code:
    from quanto import AutoQuantizer, QuantizationConfig
    config = QuantizationConfig(
        model_path="/path/to/model",
        output_dir="/output",
        precision="int4",
        layerwise=True,
    )
    quantizer = AutoQuantizer(config)
    result = quantizer.run()

    # New code:
    from quanto import UnifiedQuantizer, UnifiedConfig
    config = UnifiedConfig(
        model_path="/path/to/model",
        output_dir="/output",
        precision="int4",
        memory_strategy="lazy",  # Instead of layerwise=True
        pack_int4=True,
    )
    quantizer = UnifiedQuantizer(config)
    result = quantizer.run()
"""

from __future__ import annotations

import argparse
import sys

# Import from unified implementation
from .config import UnifiedConfig
from .unified_quantizer import UnifiedQuantizer

# Backward compatibility aliases
AutoQuantizer = UnifiedQuantizer
QuantizationConfig = UnifiedConfig


__all__ = [
    "AutoQuantizer",
    "QuantizationConfig",
    "UnifiedQuantizer",
    "UnifiedConfig",
    "main",
]


def main() -> int:
    """CLI entry point for quantization."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Quanto: Quantize a model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--model_path", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output_dir", required=True, help="Output directory for quantized model")

    # Quantization settings
    parser.add_argument(
        "--precision",
        default="mxfp4",
        choices=["int4", "int4_64", "int4_32", "int8", "fp8", "mxfp4", "mxfp6", "uint4"],
        help="Target precision",
    )
    parser.add_argument("--algorithm", default="rtn", choices=["rtn", "awq", "gptq"], help="Quantization algorithm")
    parser.add_argument("--memory_strategy", default="auto", choices=["full", "layerwise_cpu", "lazy", "auto"])

    # Sensitivity analysis
    parser.add_argument("--sensitivity_analysis", action="store_true", help="Enable iterative sensitivity analysis")
    parser.add_argument("--sensitivity_threshold", type=float, default=0.0, help="Sensitivity threshold for layer exclusion")
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="relative",
        choices=["relative", "mse", "mae", "cosine", "kl"],
        help="Metric used to rank sensitive layers",
    )
    parser.add_argument("--max_iterations", type=int, default=10, help="Max iterations for sensitivity analysis")

    # Layer exclusion
    parser.add_argument("--exclude_layers", nargs="*", help="Layer name patterns to exclude from quantization")
    parser.add_argument("--exclude_layers_file", help="JSON file containing exclude layer list")

    # Calibration data
    parser.add_argument("--calibration_data", default="pileval", help="Calibration dataset name or path")
    parser.add_argument("--num_calib_samples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for calibration")

    # Other
    parser.add_argument("--device", default="cuda", help="Device (cuda, cuda:0, cpu)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_true", help="Disable trust_remote_code")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip perplexity evaluation")
    parser.add_argument("--sensitivity_cache_on_gpu", action="store_true", default=True)

    args = parser.parse_args()

    # Handle exclude_layers from file
    exclude_layers = args.exclude_layers
    if args.exclude_layers_file:
        with open(args.exclude_layers_file) as f:
            exclude_layers = json.load(f)

    config = UnifiedConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        precision=args.precision,
        algorithm=args.algorithm,
        memory_strategy=args.memory_strategy,
        sensitivity_analysis=args.sensitivity_analysis,
        sensitivity_threshold=args.sensitivity_threshold,
        sensitivity_metric=args.sensitivity_metric,
        max_iterations=args.max_iterations,
        exclude_layers=exclude_layers,
        calibration_data=args.calibration_data,
        num_calib_samples=args.num_calib_samples,
        seq_len=args.seq_len,
        device=args.device,
        trust_remote_code=not args.no_trust_remote_code,
        skip_evaluation=args.skip_evaluation,
        sensitivity_cache_on_gpu=args.sensitivity_cache_on_gpu,
    )

    quantizer = UnifiedQuantizer(config)
    result = quantizer.run()

    if result.success:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    else:
        print(f"FAILED: {result.error_message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

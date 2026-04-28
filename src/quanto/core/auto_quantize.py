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


def main() -> int:
    """Main entry point for CLI quantization."""
    parser = argparse.ArgumentParser(
        description="Quanto: LLM Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Quantization settings
    parser.add_argument("--precision", type=str, default="int4",
                        choices=["int4", "int8", "fp8", "mxfp4", "mxfp6", "uint4"],
                        help="Quantization precision (default: int4)")
    parser.add_argument("--pack_int4", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Pack INT4 weights to INT32 (default: True)")

    # Memory strategy
    parser.add_argument("--memory_strategy", type=str, default="auto",
                        choices=["full", "layerwise_cpu", "lazy", "auto"],
                        help="Memory strategy (default: auto)")

    # Export format
    parser.add_argument("--export_format", type=str, default="quark",
                        choices=["quark", "awq", "gptq"],
                        help="Export format (default: quark)")

    # Calibration settings
    parser.add_argument("--calibration_data", type=str, default="pileval",
                        help="Calibration dataset (default: pileval)")
    parser.add_argument("--num_calib_samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length (default: 512)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")

    # Layer exclusion
    parser.add_argument("--exclude_layers", type=str, nargs="*", default=None,
                        help="Layer patterns to exclude")
    parser.add_argument("--aggressive_exclusion", action="store_true",
                        help="Use aggressive layer exclusion")

    # Sensitivity analysis
    parser.add_argument("--sensitivity_analysis", action="store_true",
                        help="Enable sensitivity analysis")
    parser.add_argument("--sensitivity_threshold", type=float, default=0.0,
                        help="Sensitivity threshold (default: 0.0, typical: 0.12-0.15)")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Max iterations for sensitivity analysis (default: 10)")

    # Other settings
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip perplexity evaluation")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Trust remote code (default: True)")

    args = parser.parse_args()

    # Handle pack_int4 (BooleanOptionalAction sets True/False directly)
    pack_int4 = args.pack_int4

    # Create config
    config = UnifiedConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        precision=args.precision,
        pack_int4=pack_int4,
        memory_strategy=args.memory_strategy,
        export_format=args.export_format,
        calibration_data=args.calibration_data,
        num_calib_samples=args.num_calib_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        exclude_layers=args.exclude_layers,
        aggressive_exclusion=args.aggressive_exclusion,
        sensitivity_analysis=args.sensitivity_analysis,
        sensitivity_threshold=args.sensitivity_threshold,
        max_iterations=args.max_iterations,
        skip_evaluation=args.skip_evaluation,
        trust_remote_code=args.trust_remote_code,
    )

    # Run quantization
    quantizer = UnifiedQuantizer(config)
    result = quantizer.run()

    if result.success:
        print(f"\nQuantization completed successfully!")
        print(f"Output: {result.output_dir}")
        if result.quantized_ppl:
            print(f"Perplexity: {result.quantized_ppl:.4f}")
        return 0
    else:
        print(f"\nQuantization failed: {result.error_message}")
        return 1


__all__ = [
    "AutoQuantizer",
    "QuantizationConfig",
    "UnifiedQuantizer",
    "UnifiedConfig",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())

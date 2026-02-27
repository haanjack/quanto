"""
Entry point for running quanto as a module.

Supports two modes:
- Quantization: python -m quanto --model_path ... --output_dir ... --precision int4
- Dequantization: python -m quanto --dequantize --model_path ... --output_dir ...
"""

import sys
import argparse


def main():
    """Main entry point that dispatches to quantize or dequantize."""
    parser = argparse.ArgumentParser(
        description="Quanto: LLM Quantization Tool",
        add_help=False,
    )

    # Add --dequantize flag to detect mode
    parser.add_argument("--dequantize", action="store_true", help="Run dequantization mode")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")

    # Parse known args to detect mode
    args, remaining = parser.parse_known_args()

    if args.dequantize:
        # Run dequantization
        from .dequantize import main as dequant_main
        # Add back --dequantize flag since dequantize module expects it
        sys.argv = [sys.argv[0], "--dequantize"] + remaining
        return dequant_main()
    else:
        # Run quantization
        from .auto_quantize import main as quant_main
        return quant_main()


if __name__ == "__main__":
    sys.exit(main())

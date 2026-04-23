"""
Entry point for running quanto as a module.

Supports two modes:
- Quantization: python -m quanto --model_path ... --output_dir ... --precision int4
- Dequantization: python -m quanto --dequantize --model_path ... --output_dir ...
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Main entry point that dispatches to quantize or dequantize."""
    # Check if --dequantize is in args
    if "--dequantize" in sys.argv:
        from quanto.core.dequantize import main as dequant_main

        return dequant_main()

    # Show top-level help only when no args or just --help with no other flags
    if len(sys.argv) <= 1 or (len(sys.argv) == 2 and sys.argv[1] in ("--help", "-h")):
        print("usage: python -m quanto [--dequantize] [options]")
        print()
        print("Quanto: LLM Quantization Tool")
        print()
        print("Modes:")
        print(
            "  Quantization:   python -m quanto --model_path ... --output_dir ... --precision mxfp4"
        )
        print("  Dequantization: python -m quanto --dequantize --model_path ... --output_dir ...")
        print()
        print("Run 'python -m quanto --model_path x --output_dir y --help' for full quantization options.")
        return 0

    # Default: quantization mode
    from quanto.core.auto_quantize import main as quant_main

    return quant_main()


if __name__ == "__main__":
    sys.exit(main())

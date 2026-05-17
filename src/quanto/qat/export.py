"""
Export utilities for QAT trials.

Wraps Quark's export_safetensors for real-quantized model export.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.torch import export_safetensors


def export_quantized_model(
    model,
    tokenizer,
    output_dir: str,
    weight_format: str = "real_quantized",
):
    """
    Export a fake-quantized model to real-quantized safetensors.

    Args:
        model: The QAT-trained model with fake quantization modules.
        tokenizer: HuggingFace tokenizer to save alongside the model.
        output_dir: Directory to write safetensors + tokenizer files.
        weight_format: "real_quantized" or "fake_quantized".
    """
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        export_safetensors(
            model=model,
            output_dir=output_dir,
            custom_mode="quark",
            weight_format=weight_format,
        )

    tokenizer.save_pretrained(output_dir)

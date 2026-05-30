"""Export utilities for QAT search results.

Wraps Quark's export_safetensors for real-quantized model export.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.torch import export_safetensors

logger = logging.getLogger(__name__)


def export_quantized_model(
    model,
    tokenizer,
    output_dir: str,
    weight_format: str = "real_quantized",
):
    """Export a fake-quantized model to real-quantized safetensors."""
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        export_safetensors(
            model=model,
            output_dir=output_dir,
            custom_mode="quark",
            weight_format=weight_format,
        )

    tokenizer.save_pretrained(output_dir)


def export_best_model(
    model_path: str,
    tokenizer_path: str | None,
    scales_path: str,
    output_dir: str,
    trust_remote_code: bool = True,
    weight_format: str = "real_quantized",
) -> str:
    """Load a fresh model, apply best QAT scales, and export as real-quantized safetensors.

    This is called after PBT search completes to produce the deployable model from
    the best member's trained scales.

    Args:
        model_path: HuggingFace model ID or path.
        tokenizer_path: Tokenizer path (defaults to model_path).
        scales_path: Path to the best member's scales.pt checkpoint.
        output_dir: Directory to write exported safetensors + tokenizer.
        trust_remote_code: Whether to trust remote code.
        weight_format: "real_quantized" or "fake_quantized".

    Returns:
        Path to the output directory.
    """
    tokenizer_path = tokenizer_path or model_path

    logger.info(f"Loading model from {model_path} for export")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    ).to("cuda:0")

    # Apply fake quantization (same as training pipeline)
    from .quantize import apply_fake_quant

    model = apply_fake_quant(
        model=model,
        tokenizer=tokenizer,
        precision="int4",
        device="cuda",
    )

    # Load the best member's trained scales
    logger.info(f"Loading trained scales from {scales_path}")
    scale_state = torch.load(scales_path, weights_only=True)
    for name, tensor in scale_state.items():
        parts = name.split(".")
        mod = model
        for part in parts[:-1]:
            mod = getattr(mod, part)
        param = getattr(mod, parts[-1])
        param.data.copy_(tensor.to(param.device))
    logger.info(f"Loaded {len(scale_state)} scale parameters")

    # Freeze quantizers (required before export_safetensors)
    logger.info("Freezing quantizers for export")
    from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

    for _name, mod in model.named_modules():
        if isinstance(mod, QuantLinear) and hasattr(mod, "_weight_quantizer"):
            wq = mod._weight_quantizer
            if hasattr(wq, "merge_scale"):
                wq.merge_scale()

    # Export
    logger.info(f"Exporting to {output_dir}")
    export_quantized_model(model, tokenizer, output_dir, weight_format)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Export complete: {output_dir}")
    return output_dir

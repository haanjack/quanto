"""
PTQ → Fake-quant step for QAT trials.

Wraps Quark's ModelQuantizer to apply fake quantization,
which is the starting point for QAT fine-tuning.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add quark to path (same as UnifiedQuantizer)
sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    Int4PerGroupSpec,
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
    Uint4PerGroupSpec,
)

from ..utils import get_calib_dataloader


def build_quant_config(
    precision: str,
    group_size: int = 128,
    symmetric: bool = True,
    exclude_layers: list[str] | None = None,
) -> QConfig:
    """
    Build Quark QConfig from QAT trial parameters.

    Args:
        precision: "int4" or "mxfp4"
        group_size: Quantization group size (128 for INT4, 32 for MXFP4)
        symmetric: Whether to use symmetric quantization (INT4 only)
        exclude_layers: Layer patterns to exclude
    """
    if precision == "int4":
        if symmetric:
            weight_spec = Int4PerGroupSpec(
                ch_axis=1,
                group_size=group_size,
            ).to_quantization_spec()
        else:
            weight_spec = Uint4PerGroupSpec(
                ch_axis=1,
                group_size=group_size,
            ).to_quantization_spec()
    elif precision == "mxfp4":
        weight_spec = OCP_MXFP4Spec(ch_axis=-1).to_quantization_spec()
    else:
        raise ValueError(f"Unsupported precision for QAT: {precision}")

    quant_layer_config = QLayerConfig(weight=weight_spec)
    return QConfig(
        global_quant_config=quant_layer_config,
        exclude=exclude_layers or ["lm_head"],
    )


def apply_fake_quant(
    model,
    tokenizer,
    precision: str,
    group_size: int = 128,
    symmetric: bool = True,
    calibration_dataset: str = "wikitext",
    num_calib_samples: int = 128,
    seq_len: int = 512,
    device: str = "cuda",
    exclude_layers: list[str] | None = None,
):
    """
    Apply fake quantization to a model (PTQ step before QAT fine-tuning).

    Returns the model with FrozenScaledFakeQuantize (INT4) or
    NonScaledFakeQuantize (MXFP4) modules inserted.
    """
    quant_config = build_quant_config(
        precision=precision,
        group_size=group_size,
        symmetric=symmetric,
        exclude_layers=exclude_layers,
    )

    calib_loader = get_calib_dataloader(
        dataset_name_or_path=calibration_dataset,
        tokenizer=tokenizer,
        num_calib_data=num_calib_samples,
        seqlen=seq_len,
        device=device,
    )

    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model, calib_loader)
    model = quantizer.freeze(model)

    return model

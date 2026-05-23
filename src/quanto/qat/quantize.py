"""
PTQ → Fake-quant step for QAT trials.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

import torch
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    Int4PerGroupSpec,
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
    Uint4PerGroupSpec,
)

from ..utils import get_calib_dataloader

logger = logging.getLogger(__name__)


def build_quant_config(
    precision: str,
    group_size: int = 128,
    symmetric: bool = True,
    exclude_layers: list[str] | None = None,
) -> QConfig:
    if precision == "int4":
        weight_spec = (Int4PerGroupSpec if symmetric else Uint4PerGroupSpec)(
            ch_axis=1, group_size=group_size,
        ).to_quantization_spec()
    elif precision == "mxfp4":
        weight_spec = OCP_MXFP4Spec(ch_axis=-1).to_quantization_spec()
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    return QConfig(
        global_quant_config=QLayerConfig(weight=weight_spec),
        exclude=exclude_layers or ["lm_head"],
    )


def apply_fake_quant(
    model, tokenizer, precision: str, group_size: int = 128, symmetric: bool = True,
    calibration_dataset: str = "wikitext", num_calib_samples: int = 128,
    seq_len: int = 512, device: str = "cuda", exclude_layers: list[str] | None = None,
):
    quant_config = build_quant_config(precision, group_size, symmetric, exclude_layers)
    calib_loader = get_calib_dataloader(
        dataset_name_or_path=calibration_dataset, tokenizer=tokenizer,
        num_calib_data=num_calib_samples, seqlen=seq_len, device=device,
    )
    quantizer = ModelQuantizer(quant_config, multi_device=True)
    model = quantizer.quantize_model(model, calib_loader)
    model = quantizer.freeze(model)

    del quantizer, calib_loader
    gc.collect()
    torch.cuda.empty_cache()
    return model

"""
Quanto: General Purpose LLM Quantization Tool
Utility modules.
"""

from __future__ import annotations

from .calibration import CalibrationDataManager, LocalTextDataset, get_calib_dataloader
from .int4_pack import (
    PACK_ORDER,
    pack_int4_to_int32,
    pack_layer_weights,
    quantize_to_int4,
    unpack_int32_to_int4,
)
from .logging import Timer, get_logger, log_with_timestamp
from .memory import (
    clear_gpu_memory,
    estimate_model_memory_gb,
    get_device_memory_gb,
    get_free_memory_gb,
    get_memory_info,
    print_memory_usage,
)
from .model_utils import (
    detect_model_type,
    get_layer_info,
    get_layer_prefix,
    get_num_layers,
    get_template,
)

__all__ = [
    # Calibration
    "CalibrationDataManager",
    "LocalTextDataset",
    "get_calib_dataloader",
    # INT4 Packing
    "PACK_ORDER",
    "quantize_to_int4",
    "pack_int4_to_int32",
    "unpack_int32_to_int4",
    "pack_layer_weights",
    # Logging
    "get_logger",
    "log_with_timestamp",
    "Timer",
    # Memory
    "clear_gpu_memory",
    "get_memory_info",
    "get_device_memory_gb",
    "get_free_memory_gb",
    "print_memory_usage",
    "estimate_model_memory_gb",
    # Model utilities
    "detect_model_type",
    "get_template",
    "get_layer_prefix",
    "get_layer_info",
    "get_num_layers",
]

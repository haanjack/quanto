"""
Quanto: General Purpose LLM Quantization Tool

A comprehensive toolkit for quantizing Large Language Models using AMD Quark.

Features:
- Multiple quantization precisions: INT8, INT4, FP8, MXFP4
- Memory-efficient layerwise quantization for large models
- Automatic layer exclusion for quality preservation
- HuggingFace model export compatibility
- Dequantization support

Basic Usage:
    from quanto import UnifiedQuantizer, UnifiedConfig

    config = UnifiedConfig(
        model_path="/path/to/model",
        output_dir="/output/path",
        precision="int4",
        memory_strategy="lazy",  # or "full", "layerwise_cpu", "auto"
    )
    quantizer = UnifiedQuantizer(config)
    result = quantizer.run()

CLI Usage:
    python -m quanto --model_path /path/to/model --precision int4 --output_dir ./output
"""

from __future__ import annotations

__version__ = "0.2.0"

# Import core classes
from .core import (
    AutoQuantizer,
    BaseQuantizer,
    QuantizationConfig,
    QuantizationResult,
    UnifiedConfig,
    UnifiedQuantizer,
)

# Always available utilities
from .constants import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_GROUP_SIZES,
    MODEL_TYPE_MAPPINGS,
    PRECISION_TO_SCHEME,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_PRECISIONS,
)
from .utils import (
    CalibrationDataManager,
    Timer,
    clear_gpu_memory,
    detect_model_type,
    get_calib_dataloader,
    get_logger,
    get_memory_info,
    get_template,
)

__all__ = [
    # Version
    "__version__",
    # Core quantization (new unified API)
    "UnifiedConfig",
    "UnifiedQuantizer",
    "QuantizationResult",
    # Core quantization (backward compatibility)
    "QuantizationConfig",
    "AutoQuantizer",
    "BaseQuantizer",
    # Constants
    "PRECISION_TO_SCHEME",
    "DEFAULT_GROUP_SIZES",
    "MODEL_TYPE_MAPPINGS",
    "DEFAULT_EXCLUDE_PATTERNS",
    "SUPPORTED_PRECISIONS",
    "SUPPORTED_ALGORITHMS",
    # Utilities
    "CalibrationDataManager",
    "get_calib_dataloader",
    "Timer",
    "get_logger",
    "clear_gpu_memory",
    "get_memory_info",
    "detect_model_type",
    "get_template",
]

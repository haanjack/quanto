"""
Quanto: General Purpose LLM Quantization Tool
Core quantization modules.
"""

from __future__ import annotations

from .base_quantizer import BaseQuantizer, QuantizationResult
from .config import UnifiedConfig
from .unified_quantizer import UnifiedQuantizer

# Backward compatibility aliases
QuantizationConfig = UnifiedConfig
AutoQuantizer = UnifiedQuantizer

__all__ = [
    # New unified API
    "UnifiedConfig",
    "UnifiedQuantizer",
    # Base classes
    "BaseQuantizer",
    "QuantizationResult",
    # Backward compatibility
    "QuantizationConfig",
    "AutoQuantizer",
]

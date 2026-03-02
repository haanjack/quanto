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

# Import from unified implementation
from .config import UnifiedConfig
from .unified_quantizer import UnifiedQuantizer

# Backward compatibility aliases
AutoQuantizer = UnifiedQuantizer
QuantizationConfig = UnifiedConfig

__all__ = [
    "AutoQuantizer",
    "QuantizationConfig",
    "UnifiedQuantizer",
    "UnifiedConfig",
]

"""
Unified Configuration for Quantization.

This module provides a single configuration class that consolidates all settings
from the previous quantizer implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# Supported export formats
ExportFormat = Literal["quark", "awq", "gptq"]


@dataclass
class UnifiedConfig:
    """
    Unified configuration for all quantization strategies.

    This configuration supports:
    - Full GPU quantization (load entire model to GPU)
    - Layerwise CPU quantization (load to CPU, quantize layers on GPU)
    - Lazy quantization (load weights on-demand from disk)
    - Automatic strategy selection based on model size vs GPU memory

    Attributes:
        model_path: Path to the model directory
        output_dir: Output directory for quantized model
        precision: Target precision (int4, int8, fp8, mxfp4)
        pack_int4: Pack INT4 weights to INT32 for storage efficiency
        memory_strategy: Memory strategy ("full", "layerwise_cpu", "lazy", "auto")
        export_format: Export format ("quark", "awq", "gptq") for vLLM compatibility
        calibration_data: Calibration dataset name or path
        num_calib_samples: Number of calibration samples
        seq_len: Sequence length for calibration
        batch_size: Batch size for calibration
        device: Device to use for computation
        exclude_layers: Layer patterns to exclude from quantization
        aggressive_exclusion: Use aggressive layer exclusion rules
        sensitivity_analysis: Enable sequential sensitivity analysis for layer exclusion
        sensitivity_threshold: Threshold for excluding sensitive layers
        sensitivity_cache_on_gpu: Cache activations on GPU (faster, more memory)
        skip_evaluation: Skip perplexity evaluation
        trust_remote_code: Trust remote code when loading models
        debug_dir: Directory for debug output
    """

    # Required settings
    model_path: str
    output_dir: str

    # Quantization settings
    precision: str = "int4"  # int4, int8, fp8, mxfp4
    pack_int4: bool = True  # Pack INT4 to INT32 (only for int4 precision)

    # Memory strategy: "full", "layerwise_cpu", "lazy", "auto"
    memory_strategy: str = "auto"

    # Export format: "quark", "awq", "gptq"
    # - "quark": Native Quark format (default)
    # - "awq": AWQ format for vLLM compatibility (uses qweight, scales, qzeros)
    # - "gptq": GPTQ format for vLLM compatibility (uses qweight, scales, g_idx)
    export_format: ExportFormat = "quark"

    # Calibration settings
    calibration_data: str = "pileval"
    num_calib_samples: int = 128
    seq_len: int = 512
    batch_size: int = 1

    # Device settings
    device: str = "cuda"

    # Layer exclusion settings
    exclude_layers: list[str] | None = None
    aggressive_exclusion: bool = False

    # Sensitivity-based exclusion
    sensitivity_analysis: bool = False  # Enable sequential sensitivity analysis
    sensitivity_threshold: float = 0.0  # Threshold for excluding sensitive layers (0 = disabled, typical values: 0.12-0.15 for INT4)
    sensitivity_cache_on_gpu: bool = True  # Cache activations on GPU (faster, uses more memory)

    # Evaluation settings
    skip_evaluation: bool = False

    # Other settings
    trust_remote_code: bool = True
    debug_dir: str = "./debug_output"

    # Layer batch size for lazy mode (number of layers to process in parallel)
    layer_batch_size: int = 4

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration for conflicting options."""
        # Validate precision
        valid_precisions = ["int4", "int8", "fp8", "mxfp4", "mxfp6", "uint4", "int4_64", "int4_32"]
        if self.precision not in valid_precisions:
            raise ValueError(
                f"Invalid precision '{self.precision}'. Must be one of: {valid_precisions}"
            )

        # Validate memory strategy
        valid_strategies = ["full", "layerwise_cpu", "lazy", "auto"]
        if self.memory_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid memory_strategy '{self.memory_strategy}'. Must be one of: {valid_strategies}"
            )

        # Validate export format
        valid_formats = ["quark", "awq", "gptq"]
        if self.export_format not in valid_formats:
            raise ValueError(
                f"Invalid export_format '{self.export_format}'. Must be one of: {valid_formats}"
            )

        # AWQ/GPTQ export only works with INT4
        if self.export_format in ["awq", "gptq"] and not self.precision.startswith("int4"):
            raise ValueError(
                f"export_format '{self.export_format}' only supports INT4 precision, "
                f"got precision '{self.precision}'"
            )

        # Pack INT4 only makes sense for int4 precision
        if self.pack_int4 and not self.precision.startswith("int4"):
            self.pack_int4 = False  # Silently disable for non-INT4 precisions

        # Validate batch sizes
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.layer_batch_size < 1:
            raise ValueError(f"layer_batch_size must be >= 1, got {self.layer_batch_size}")

        # Validate calibration samples
        if self.num_calib_samples < 1:
            raise ValueError(f"num_calib_samples must be >= 1, got {self.num_calib_samples}")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "precision": self.precision,
            "pack_int4": self.pack_int4,
            "memory_strategy": self.memory_strategy,
            "export_format": self.export_format,
            "calibration_data": self.calibration_data,
            "num_calib_samples": self.num_calib_samples,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "device": self.device,
            "exclude_layers": self.exclude_layers,
            "aggressive_exclusion": self.aggressive_exclusion,
            "sensitivity_analysis": self.sensitivity_analysis,
            "sensitivity_threshold": self.sensitivity_threshold,
            "sensitivity_cache_on_gpu": self.sensitivity_cache_on_gpu,
            "skip_evaluation": self.skip_evaluation,
            "trust_remote_code": self.trust_remote_code,
            "debug_dir": self.debug_dir,
            "layer_batch_size": self.layer_batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnifiedConfig:
        """Create configuration from dictionary."""
        return cls(**data)


# Backward compatibility alias
QuantizationConfig = UnifiedConfig

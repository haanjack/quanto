"""
Quanto: General Purpose LLM Quantization Tool
Base class for all quantizers.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..constants import PRECISION_TO_SCHEME, SUPPORTED_PRECISIONS
from ..utils import (
    Timer,
    clear_gpu_memory,
    detect_model_type,
    get_logger,
    get_memory_info,
    get_template,
)

if TYPE_CHECKING:
    from quark.torch import LLMTemplate


@dataclass
class QuantizationResult:
    """Result of quantization process."""

    success: bool
    output_dir: str | None = None
    original_ppl: float | None = None
    quantized_ppl: float | None = None
    ppl_change: float | None = None
    exclude_layers_used: list[str] = field(default_factory=list)
    model_type: str | None = None
    quant_scheme: str | None = None
    precision: str | None = None
    error_message: str | None = None
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "original_ppl": self.original_ppl,
            "quantized_ppl": self.quantized_ppl,
            "ppl_change": self.ppl_change,
            "exclude_layers_used": self.exclude_layers_used,
            "model_type": self.model_type,
            "quant_scheme": self.quant_scheme,
            "precision": self.precision,
            "error_message": self.error_message,
            "timing": self.timing,
        }


class BaseQuantizer(ABC):
    """
    Abstract base class for all quantizers.

    Provides common functionality:
    - Model loading and detection
    - Template selection
    - Logging utilities
    - Memory management
    - Result handling
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        precision: str = "int4",
        device: str = "cuda",
        trust_remote_code: bool = True,
    ):
        """
        Initialize base quantizer.

        Args:
            model_path: Path to the model directory
            output_dir: Output directory for quantized model
            precision: Target precision (int4, int8, fp8, etc.)
            device: Device to use for computation
            trust_remote_code: Whether to trust remote code
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.precision = precision
        self.device = device
        self.trust_remote_code = trust_remote_code

        # Validate precision
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision: {precision}. Supported: {SUPPORTED_PRECISIONS}"
            )

        # State
        self.model: nn.Module | None = None
        self.tokenizer: Any | None = None
        self.model_type: str | None = None
        self.template: LLMTemplate | None = None
        self.timing: dict[str, float] = {}

        # Logger
        self.logger = get_logger(self.__class__.__name__)

    def _log(self, message: str) -> None:
        """Print log message with timestamp (for backward compatibility)."""
        self.logger.info(message)

    def _get_quant_scheme(self) -> str:
        """Get quantization scheme from precision setting."""
        return PRECISION_TO_SCHEME.get(self.precision, self.precision)

    def detect_model_type(self) -> str:
        """Detect model type from model path."""
        self.model_type = detect_model_type(self.model_path, self.trust_remote_code)
        self._log(f"Detected model type: {self.model_type}")
        return self.model_type

    def get_template(self) -> LLMTemplate | None:
        """Get LLMTemplate for the model."""
        if self.model_type is None:
            self.detect_model_type()

        self.template = get_template(self.model_type)
        if self.template:
            self._log(f"Using template: {self.template.model_type}")
        else:
            self._log(f"Warning: No template found for model type '{self.model_type}'")

        return self.template

    def load_model(
        self,
        memory_efficient: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Load model and tokenizer.

        Args:
            memory_efficient: Use device_map="auto" for memory efficient loading
            dtype: Data type for weights (default: bfloat16 if supported, else float16)
        """
        with Timer("Model loading", self.logger) as t:
            # Detect model type
            self.detect_model_type()

            # Get template
            self.get_template()

            # Default dtype
            if dtype is None:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            # Load tokenizer
            self._log("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self._log("Loading model...")
            if memory_efficient:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    trust_remote_code=self.trust_remote_code,
                ).to(self.device)

            self.model.eval()

        self.timing["model_loading"] = t.elapsed

    def clear_memory(self) -> None:
        """Clear GPU memory."""
        clear_gpu_memory()

    def get_memory_info(self) -> str:
        """Get current memory usage."""
        return get_memory_info()

    def save_result(self, result: QuantizationResult) -> Path:
        """
        Save quantization result to JSON file.

        Args:
            result: Quantization result to save

        Returns:
            Path to saved result file
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result_path = output_path / "quantization_result.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self._log(f"Results saved to {result_path}")
        return result_path

    def print_summary(self, result: QuantizationResult) -> None:
        """Print quantization summary."""
        print("\n" + "=" * 60)
        print("QUANTIZATION SUMMARY")
        print("=" * 60)
        print(f"Model Type: {result.model_type}")
        print(f"Precision: {result.precision}")
        print(f"Quantization Scheme: {result.quant_scheme}")
        print(f"Output Directory: {result.output_dir}")
        print(f"\nExcluded Layers: {result.exclude_layers_used}")

        if result.original_ppl is not None:
            print(f"\nOriginal PPL: {result.original_ppl:.4f}")
        if result.quantized_ppl is not None:
            print(f"Quantized PPL: {result.quantized_ppl:.4f}")
        if result.ppl_change is not None:
            print(f"PPL Change: {result.ppl_change:+.4f}")

        if result.timing:
            print("\nTiming:")
            for stage, duration in result.timing.items():
                print(f"  {stage}: {duration:.2f}s")

        print("=" * 60)

    @abstractmethod
    def run(self) -> QuantizationResult:
        """
        Run the quantization process.

        Returns:
            QuantizationResult with details of the quantization
        """
        pass

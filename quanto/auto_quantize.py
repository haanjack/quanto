"""
Quanto: General Purpose LLM Quantization Tool

This module provides an automated way to quantize LLM models with minimal manual configuration.
It automatically:
- Detects model architecture and selects appropriate templates
- Identifies sensitive layers to exclude
- Loads calibration data
- Performs quantization
- Evaluates model quality before and after quantization
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

# Add quark to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.torch import (
    LLMTemplate,
    ModelQuantizer,
    export_safetensors,
)
from quark.torch.quantization.config.config import QConfig
from quark.contrib.llm_eval import ppl_eval

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

from .layer_analyzer import LayerAnalyzer, LayerAnalysisResult
from .calibration import CalibrationDataManager, get_calib_dataloader
from .layerwise_quant import LayerwiseQuantizer


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    model_path: str
    output_dir: str
    precision: str = "int8"  # int8, int4, fp8, mxfp4
    calibration_data: str = "pileval"
    num_calib_samples: int = 128
    seq_len: int = 512
    batch_size: int = 1
    device: str = "cuda"
    exclude_layers: list[str] | None = None
    exclude_last_n_layers: int = 0
    aggressive_exclusion: bool = False
    custom_exclude_patterns: list[str] | None = None
    skip_evaluation: bool = False
    memory_efficient: bool = False
    layerwise: bool = False  # True layer-wise quantization for large models
    trust_remote_code: bool = True
    model_export: str = "hf_format"

    # Advanced options
    quant_algo: str | None = None  # awq, gptq, smoothquant
    group_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "precision": self.precision,
            "calibration_data": self.calibration_data,
            "num_calib_samples": self.num_calib_samples,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "device": self.device,
            "exclude_layers": self.exclude_layers,
            "exclude_last_n_layers": self.exclude_last_n_layers,
            "aggressive_exclusion": self.aggressive_exclusion,
            "custom_exclude_patterns": self.custom_exclude_patterns,
            "skip_evaluation": self.skip_evaluation,
            "memory_efficient": self.memory_efficient,
            "layerwise": self.layerwise,
            "trust_remote_code": self.trust_remote_code,
            "model_export": self.model_export,
            "quant_algo": self.quant_algo,
            "group_size": self.group_size,
        }


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
    error_message: str | None = None
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "original_ppl": self.original_ppl,
            "quantized_ppl": self.quantized_ppl,
            "ppl_change": self.ppl_change,
            "exclude_layers_used": self.exclude_layers_used,
            "model_type": self.model_type,
            "quant_scheme": self.quant_scheme,
            "error_message": self.error_message,
            "timing": self.timing,
        }


class AutoQuantizer:
    """
    Automatic LLM Quantizer.

    This class provides an automated pipeline for quantizing LLM models
    with intelligent layer exclusion and quality evaluation.
    """

    # Mapping from precision names to Quark schemes
    # Note: For LLMs, weight-only quantization is typically preferred
    # INT8 in Quark includes activation quantization which may cause degradation
    # For better quality, use INT4 weight-only or FP8
    PRECISION_TO_SCHEME = {
        "int8": "int8",  # INT8 weight + activation (may cause degradation)
        "int4": "int4_wo_128",  # INT4 weight-only with group size 128 (recommended)
        "int4_64": "int4_wo_64",  # INT4 weight-only with group size 64
        "int4_32": "int4_wo_32",  # INT4 weight-only with group size 32
        "fp8": "fp8",  # FP8 (weight-only)
        "mxfp4": "mxfp4",  # MXFP4
        "mxfp6": "mxfp6_e3m2",  # MXFP6
        "uint4": "uint4_wo_128",  # UINT4 weight-only
    }

    def __init__(self, config: QuantizationConfig):
        """
        Initialize AutoQuantizer.

        Args:
            config: Quantization configuration
        """
        self.config = config
        self.model: nn.Module | None = None
        self.tokenizer: Any = None
        self.model_type: str | None = None
        self.template: LLMTemplate | None = None
        self.timing: dict[str, float] = {}

    def _log(self, message: str) -> None:
        """Print log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _detect_model_type(self) -> str:
        """Detect model type from model config."""
        config_path = Path(self.config.model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type", config.get("architectures", ["unknown"])[0])
        else:
            # Load config from model
            config = AutoConfig.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            model_type = getattr(config, "model_type", getattr(config, "architectures", ["unknown"])[0])

        return model_type

    def _get_template(self, model_type: str) -> LLMTemplate | None:
        """Get LLMTemplate for model type."""
        available_templates = LLMTemplate.list_available()

        # Try exact match first
        if model_type in available_templates:
            return LLMTemplate.get(model_type)

        # Try common mappings
        type_mappings = {
            "llama": "llama",
            "llama3": "llama",
            "qwen2": "qwen2",
            "qwen3": "qwen3",
            "qwen": "qwen",
            "mistral": "mistral",
            "mixtral": "mixtral",
            "deepseek": "deepseek",
            "gemma": "gemma2",
            "gemma2": "gemma2",
            "phi": "phi",
            "phi3": "phi3",
        }

        for key, template_name in type_mappings.items():
            if key in model_type.lower():
                if template_name in available_templates:
                    return LLMTemplate.get(template_name)

        # Try partial match
        for template_name in available_templates:
            if model_type.lower() in template_name.lower() or template_name.lower() in model_type.lower():
                return LLMTemplate.get(template_name)

        self._log(f"Warning: No template found for model type '{model_type}'")
        self._log(f"Available templates: {available_templates}")
        return None

    def _get_quant_scheme(self) -> str:
        """Get quantization scheme from precision setting."""
        scheme = self.PRECISION_TO_SCHEME.get(self.config.precision, self.config.precision)

        # Validate scheme is available
        available_schemes = LLMTemplate.get_supported_schemes()
        if scheme not in available_schemes:
            self._log(f"Warning: Scheme '{scheme}' not in available schemes: {available_schemes}")
            # Fall back to int8
            scheme = "int8"

        return scheme

    def load_model(self) -> None:
        """Load model and tokenizer."""
        start_time = time.time()
        self._log(f"Loading model from {self.config.model_path}...")

        # Detect model type
        self.model_type = self._detect_model_type()
        self._log(f"Detected model type: {self.model_type}")

        # Get template
        self.template = self._get_template(self.model_type)
        if self.template:
            self._log(f"Using template: {self.template.model_type}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if self.config.memory_efficient:
            # Use device_map="auto" for memory efficient loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=self.config.trust_remote_code,
            ).to(self.config.device)

        self.model.eval()
        self.timing["model_loading"] = time.time() - start_time
        self._log(f"Model loaded in {self.timing['model_loading']:.2f}s")

    def analyze_layers(self) -> LayerAnalysisResult:
        """Analyze model layers to determine exclusion patterns."""
        start_time = time.time()
        self._log("Analyzing model layers...")

        analyzer = LayerAnalyzer(
            model=self.model,
            model_type=self.model_type,
            template=self.template,
        )

        result = analyzer.analyze(
            aggressive=self.config.aggressive_exclusion,
            exclude_last_n_layers=self.config.exclude_last_n_layers,
            custom_patterns=self.config.custom_exclude_patterns,
        )

        self.timing["layer_analysis"] = time.time() - start_time
        self._log(f"Layer analysis completed in {self.timing['layer_analysis']:.2f}s")
        self._log(f"Recommended exclusion patterns: {result.exclude_patterns}")

        return result

    def get_calibration_dataloader(self) -> Any:
        """Get calibration dataloader."""
        start_time = time.time()
        self._log(f"Loading calibration data from {self.config.calibration_data}...")

        calib_loader = get_calib_dataloader(
            dataset_name_or_path=self.config.calibration_data,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            num_calib_data=self.config.num_calib_samples,
            seqlen=self.config.seq_len,
            device=str(self.model.device) if self.model else self.config.device,
        )

        self.timing["calibration_loading"] = time.time() - start_time
        self._log(f"Calibration data loaded in {self.timing['calibration_loading']:.2f}s")

        return calib_loader

    def evaluate_model(self, model: nn.Module | None = None) -> float:
        """Evaluate model perplexity on wikitext-2."""
        start_time = time.time()
        self._log("Evaluating model perplexity...")

        model_to_eval = model or self.model
        device = str(model_to_eval.device)

        # Load wikitext-2 test data
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = self.tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

        # Evaluate
        ppl = ppl_eval(model_to_eval, testenc, device)

        self.timing["evaluation"] = time.time() - start_time
        self._log(f"Perplexity: {ppl.item():.4f} (evaluated in {self.timing['evaluation']:.2f}s)")

        return ppl.item()

    def quantize(self, exclude_layers: list[str] | None = None) -> None:
        """
        Quantize the model.

        Args:
            exclude_layers: List of layer patterns to exclude from quantization
        """
        start_time = time.time()
        self._log("Starting quantization...")

        # Get quantization scheme
        quant_scheme = self._get_quant_scheme()
        self._log(f"Using quantization scheme: {quant_scheme}")

        # Get calibration dataloader
        calib_loader = self.get_calibration_dataloader()

        # Prepare exclude layers
        if exclude_layers is None:
            analysis = self.analyze_layers()
            exclude_layers = analysis.exclude_patterns

        self._log(f"Excluding layers: {exclude_layers}")

        # Build quantization config
        if self.template:
            quant_config = self.template.get_config(
                scheme=quant_scheme,
                algorithm=self.config.quant_algo,
                exclude_layers=exclude_layers,
            )
        else:
            # Create a basic config without template
            from quark.torch.quantization.config.config import QConfig, QLayerConfig, Int8PerTensorSpec

            quant_config = QConfig(
                global_quant_config=QLayerConfig(weight=Int8PerTensorSpec().to_quantization_spec()),
                exclude=exclude_layers,
            )

        # Create quantizer
        quantizer = ModelQuantizer(quant_config, multi_device=self.config.memory_efficient)

        # Quantize model
        self._log("Quantizing model...")
        self.model = quantizer.quantize_model(self.model, calib_loader)

        # Freeze model
        self._log("Freezing quantized model...")
        self.model = quantizer.freeze(self.model)

        self.timing["quantization"] = time.time() - start_time
        self._log(f"Quantization completed in {self.timing['quantization']:.2f}s")

    def export_model(self) -> None:
        """Export quantized model."""
        start_time = time.time()
        self._log(f"Exporting model to {self.config.output_dir}...")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Export model
        with torch.no_grad():
            export_safetensors(
                model=self.model,
                output_dir=self.config.output_dir,
                custom_mode="quark",
                weight_format="real_quantized",
            )

        # Save tokenizer
        self.tokenizer.save_pretrained(self.config.output_dir)

        # Save quantization config
        config_path = Path(self.config.output_dir) / "quantization_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self.timing["export"] = time.time() - start_time
        self._log(f"Model exported in {self.timing['export']:.2f}s")

    def run(self) -> QuantizationResult:
        """
        Run the complete quantization pipeline.

        Returns:
            QuantizationResult with details of the quantization
        """
        result = QuantizationResult(success=False)

        # Check if layerwise mode is enabled
        if self.config.layerwise:
            return self._run_layerwise_quantization()

        try:
            # Load model
            self.load_model()

            # Evaluate original model
            original_ppl = None
            if not self.config.skip_evaluation:
                original_ppl = self.evaluate_model()
                result.original_ppl = original_ppl

            # Analyze layers
            analysis = self.analyze_layers()

            # Determine exclude layers
            exclude_layers = self.config.exclude_layers or analysis.exclude_patterns
            result.exclude_layers_used = exclude_layers

            # Quantize
            self.quantize(exclude_layers)

            # Evaluate quantized model
            quantized_ppl = None
            if not self.config.skip_evaluation:
                quantized_ppl = self.evaluate_model()
                result.quantized_ppl = quantized_ppl
                result.ppl_change = quantized_ppl - original_ppl if original_ppl else None

            # Export
            self.export_model()

            result.success = True
            result.output_dir = self.config.output_dir
            result.model_type = self.model_type
            result.quant_scheme = self._get_quant_scheme()
            result.timing = self.timing

            # Print summary
            self._print_summary(result)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._log(f"Error during quantization: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _print_summary(self, result: QuantizationResult) -> None:
        """Print quantization summary."""
        print("\n" + "=" * 60)
        print("QUANTIZATION SUMMARY")
        print("=" * 60)
        print(f"Model Type: {result.model_type}")
        print(f"Quantization Scheme: {result.quant_scheme}")
        print(f"Output Directory: {result.output_dir}")
        print(f"\nExcluded Layers: {result.exclude_layers_used}")

        if result.original_ppl is not None:
            print(f"\nOriginal PPL: {result.original_ppl:.4f}")
        if result.quantized_ppl is not None:
            print(f"Quantized PPL: {result.quantized_ppl:.4f}")
        if result.ppl_change is not None:
            print(f"PPL Change: {result.ppl_change:+.4f}")

        print(f"\nTiming:")
        for stage, duration in result.timing.items():
            print(f"  {stage}: {duration:.2f}s")

        print("=" * 60)

    def _run_layerwise_quantization(self) -> QuantizationResult:
        """
        Run layer-wise quantization for large models.

        Returns:
            QuantizationResult with details of the quantization
        """
        result = QuantizationResult(success=False)

        try:
            # Determine exclude layers if not specified
            exclude_layers = self.config.exclude_layers
            if exclude_layers is None:
                # Default exclusions
                exclude_layers = ["lm_head"]
                if self.config.aggressive_exclusion:
                    exclude_layers.extend(["*gate*", "*embed*", "*norm*"])

            # Create layerwise quantizer
            self._log("Using layer-wise quantization mode for large models...")
            layerwise = LayerwiseQuantizer(
                model_path=self.config.model_path,
                output_dir=self.config.output_dir,
                precision=self.config.precision,
                calibration_data=self.config.calibration_data,
                num_calib_samples=self.config.num_calib_samples,
                seq_len=self.config.seq_len,
                batch_size=self.config.batch_size,
                device=self.config.device,
                exclude_layers=exclude_layers,
                trust_remote_code=self.config.trust_remote_code,
            )

            # Run quantization using the GPU-pipelined method (GPU-only computation)
            quant_result = layerwise.run_pipeline_quantization()

            result.success = quant_result.get("success", False)
            result.output_dir = quant_result.get("output_dir")
            result.model_type = quant_result.get("model_type")
            result.quant_scheme = quant_result.get("quant_scheme")
            result.exclude_layers_used = exclude_layers
            result.timing = quant_result.get("timing", {})

            self._print_summary(result)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._log(f"Error during layer-wise quantization: {e}")
            import traceback
            traceback.print_exc()

        return result


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Auto-Quantize: General Purpose LLM Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize Llama-3-8B to INT8
  python -m auto_quantize --model_path ~/models/meta-llama/Meta-Llama-3-8B --precision int8 --output_dir ./quantized/llama3-int8

  # Quantize Qwen3-32B to INT4 with custom calibration data
  python -m auto_quantize --model_path ~/models/qwen/qwen3-32b --precision int4 --calibration_data ~/datasets/mit-han-lab/pile-val-backup --output_dir ./quantized/qwen3-int4

  # Use aggressive layer exclusion for better quality
  python -m auto_quantize --model_path ~/models/meta-llama/Meta-Llama-3-8B --precision int4 --aggressive_exclusion --output_dir ./quantized/llama3-int4
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )

    # Quantization options
    parser.add_argument(
        "--precision",
        type=str,
        default="int8",
        choices=["int8", "int4", "int4_64", "int4_32", "fp8", "mxfp4", "mxfp6", "uint4"],
        help="Target precision for quantization (default: int8)",
    )
    parser.add_argument(
        "--quant_algo",
        type=str,
        default=None,
        choices=["awq", "gptq", "smoothquant"],
        help="Optional quantization algorithm",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="Group size for group-wise quantization",
    )

    # Calibration options
    parser.add_argument(
        "--calibration_data",
        type=str,
        default="pileval",
        help="Calibration dataset name or local path (default: pileval)",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence length for calibration (default: 512)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration (default: 1)",
    )

    # Layer exclusion options
    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="*",
        default=None,
        help="Layer patterns to exclude from quantization",
    )
    parser.add_argument(
        "--exclude_last_n_layers",
        type=int,
        default=0,
        help="Number of last transformer layers to exclude (default: 0 = auto)",
    )
    parser.add_argument(
        "--aggressive_exclusion",
        action="store_true",
        help="Use aggressive layer exclusion for better quality",
    )

    # Other options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory-efficient model loading",
    )
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="Use layer-wise quantization for large models (loads one layer at a time)",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip perplexity evaluation",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading models",
    )

    args = parser.parse_args()

    # Create config
    config = QuantizationConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        precision=args.precision,
        calibration_data=args.calibration_data,
        num_calib_samples=args.num_calib_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        exclude_layers=args.exclude_layers,
        exclude_last_n_layers=args.exclude_last_n_layers,
        aggressive_exclusion=args.aggressive_exclusion,
        skip_evaluation=args.skip_evaluation,
        memory_efficient=args.memory_efficient,
        layerwise=args.layerwise,
        trust_remote_code=args.trust_remote_code,
        quant_algo=args.quant_algo,
        group_size=args.group_size,
    )

    # Run quantization
    quantizer = AutoQuantizer(config)
    result = quantizer.run()

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = Path(args.output_dir) / "quantization_result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {result_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

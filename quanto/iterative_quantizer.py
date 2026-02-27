"""
Iterative Quantization Pipeline: Automatically find optimal layer exclusions.

This module implements an iterative process that:
1. Runs initial quantization with minimal exclusions
2. Collects layer diagnostics
3. Identifies sensitive layers
4. Updates exclusion list
5. Re-runs quantization
6. Evaluates quality (perplexity)
7. Iterates until quality criteria are met
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

# Add quark to path
sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
from quark.torch.quantization.config.config import QConfig, QLayerConfig, Int4PerGroupSpec
from quark.contrib.llm_eval import ppl_eval

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

# Handle both relative and absolute imports
try:
    from .sensitivity_analyzer import (
        SensitivityAnalyzer,
        SensitivityAnalysisResult,
        convert_exclusions_to_patterns,
    )
    from .calibration import get_calib_dataloader
except ImportError:
    from sensitivity_analyzer import (
        SensitivityAnalyzer,
        SensitivityAnalysisResult,
        convert_exclusions_to_patterns,
    )
    from calibration import get_calib_dataloader


@dataclass
class IterativeConfig:
    """Configuration for iterative quantization."""

    # Model settings
    model_path: str
    output_dir: str
    precision: str = "int4"  # int4, int8, fp8

    # Calibration settings
    calibration_data: str = "pileval"
    num_calib_samples: int = 128
    seq_len: int = 512
    batch_size: int = 1

    # Device settings
    device: str = "cuda"
    memory_efficient: bool = False

    # Iteration settings
    max_iterations: int = 5
    target_ppl_degradation: float = 2.0  # Stop when PPL increase < this value
    min_ppl_degradation: float = 0.1  # Minimum improvement to continue

    # Exclusion settings
    initial_exclusions: list[str] | None = None
    max_exclude_percent: float = 20.0  # Max % of layers to exclude

    # Advanced settings
    trust_remote_code: bool = True
    skip_evaluation: bool = False
    debug_dir: str = "./debug_output"

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
            "memory_efficient": self.memory_efficient,
            "max_iterations": self.max_iterations,
            "target_ppl_degradation": self.target_ppl_degradation,
            "min_ppl_degradation": self.min_ppl_degradation,
            "initial_exclusions": self.initial_exclusions,
            "max_exclude_percent": self.max_exclude_percent,
            "trust_remote_code": self.trust_remote_code,
            "skip_evaluation": self.skip_evaluation,
            "debug_dir": self.debug_dir,
        }


@dataclass
class IterationResult:
    """Result of a single iteration."""

    iteration: int
    exclude_layers: list[str]
    original_ppl: float | None = None
    quantized_ppl: float | None = None
    ppl_degradation: float | None = None
    sensitivity_analysis: SensitivityAnalysisResult | None = None
    success: bool = False
    error_message: str | None = None
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "exclude_layers": self.exclude_layers,
            "original_ppl": self.original_ppl,
            "quantized_ppl": self.quantized_ppl,
            "ppl_degradation": self.ppl_degradation,
            "sensitivity_analysis": self.sensitivity_analysis.to_dict() if self.sensitivity_analysis else None,
            "success": self.success,
            "error_message": self.error_message,
            "timing": self.timing,
        }


@dataclass
class IterativeResult:
    """Final result of iterative quantization."""

    success: bool
    output_dir: str | None = None
    best_iteration: int = 0
    best_ppl_degradation: float | None = None
    final_exclude_layers: list[str] = field(default_factory=list)
    iterations: list[IterationResult] = field(default_factory=list)
    model_type: str | None = None
    total_timing: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "best_iteration": self.best_iteration,
            "best_ppl_degradation": self.best_ppl_degradation,
            "final_exclude_layers": self.final_exclude_layers,
            "iterations": [i.to_dict() for i in self.iterations],
            "model_type": self.model_type,
            "total_timing": self.total_timing,
            "error_message": self.error_message,
        }


class IterativeQuantizer:
    """
    Iterative quantization pipeline that automatically finds optimal exclusions.
    """

    PRECISION_TO_SCHEME = {
        "int8": "int8",
        "int4": "int4_wo_128",
        "int4_64": "int4_wo_64",
        "int4_32": "int4_wo_32",
        "fp8": "fp8",
    }

    def __init__(self, config: IterativeConfig):
        self.config = config
        self.model: nn.Module | None = None
        self.tokenizer: Any = None
        self.model_type: str | None = None
        self.template: LLMTemplate | None = None
        self.original_ppl: float | None = None
        self.timing: dict[str, float] = {}

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _detect_model_type(self) -> str:
        config_path = Path(self.config.model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            return config.get("model_type", config.get("architectures", ["unknown"])[0])
        else:
            config = AutoConfig.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            return getattr(config, "model_type", getattr(config, "architectures", ["unknown"])[0])

    def _get_template(self, model_type: str) -> LLMTemplate | None:
        available_templates = LLMTemplate.list_available()

        if model_type in available_templates:
            return LLMTemplate.get(model_type)

        type_mappings = {
            "llama": "llama", "llama3": "llama",
            "qwen2": "qwen2", "qwen3": "qwen3", "qwen": "qwen",
            "mistral": "mistral", "mixtral": "mixtral",
            "deepseek": "deepseek",
            "gemma": "gemma2", "gemma2": "gemma2",
            "phi": "phi", "phi3": "phi3",
        }

        for key, template_name in type_mappings.items():
            if key in model_type.lower():
                if template_name in available_templates:
                    return LLMTemplate.get(template_name)

        return None

    def load_model(self) -> None:
        start_time = time.time()
        self._log(f"Loading model from {self.config.model_path}...")

        self.model_type = self._detect_model_type()
        self._log(f"Detected model type: {self.model_type}")

        self.template = self._get_template(self.model_type)
        if self.template:
            self._log(f"Using template: {self.template.model_type}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.config.memory_efficient:
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

    def evaluate_ppl(self, model: nn.Module | None = None, max_seq_len: int = 2048) -> float:
        start_time = time.time()
        self._log("Evaluating model perplexity...")

        model_to_eval = model or self.model
        device = str(model_to_eval.device)

        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(testdata["text"])
        testenc = self.tokenizer(text, return_tensors="pt")

        # Truncate to max sequence length to avoid context window issues
        if testenc.input_ids.shape[1] > max_seq_len:
            self._log(f"Truncating test data from {testenc.input_ids.shape[1]} to {max_seq_len} tokens")
            testenc = {k: v[:, :max_seq_len] for k, v in testenc.items()}

        ppl = ppl_eval(model_to_eval, testenc, device)

        self.timing["evaluation"] = time.time() - start_time
        self._log(f"Perplexity: {ppl.item():.4f} (evaluated in {self.timing['evaluation']:.2f}s)")

        return ppl.item()

    def run_calibration_with_debug(
        self,
        exclude_layers: list[str],
        debug_dir: str,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """
        Run calibration with debug statistics collection.

        Returns:
            Tuple of (calibrated model, debug statistics dict)
        """
        start_time = time.time()
        self._log(f"Running calibration with debug output to {debug_dir}...")

        # Setup debug environment
        os.makedirs(debug_dir, exist_ok=True)
        os.environ["QUARK_DEBUG"] = debug_dir
        os.environ["QUARK_CHECK_SCALE"] = "1"

        # Get calibration dataloader
        calib_loader = get_calib_dataloader(
            dataset_name_or_path=self.config.calibration_data,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            num_calib_data=self.config.num_calib_samples,
            seqlen=self.config.seq_len,
            device=str(self.model.device) if self.model else self.config.device,
        )

        # Build quantization config
        quant_scheme = self.PRECISION_TO_SCHEME.get(self.config.precision, "int4_wo_128")

        if self.template:
            quant_config = self.template.get_config(
                scheme=quant_scheme,
                exclude_layers=exclude_layers,
            )
        else:
            from quark.torch.quantization.config.config import Int8PerTensorSpec
            quant_config = QConfig(
                global_quant_config=QLayerConfig(weight=Int8PerTensorSpec().to_quantization_spec()),
                exclude=exclude_layers,
            )

        # Create quantizer and calibrate
        quantizer = ModelQuantizer(quant_config, multi_device=self.config.memory_efficient)
        model = quantizer.quantize_model(self.model, calib_loader)

        # Collect debug statistics (they're saved to debug_dir by Quark)
        debug_stats = self._collect_debug_stats(debug_dir)

        self.timing["calibration"] = time.time() - start_time
        self._log(f"Calibration completed in {self.timing['calibration']:.2f}s")

        # Clean up environment
        del os.environ["QUARK_DEBUG"]
        del os.environ["QUARK_CHECK_SCALE"]

        return model, debug_stats

    def _collect_debug_stats(self, debug_dir: str) -> dict[str, Any]:
        """Collect debug statistics from Quark output files."""
        debug_stats = {}
        debug_path = Path(debug_dir)

        # Load per-layer stats
        for json_file in debug_path.glob("*_stats.json"):
            try:
                with open(json_file) as f:
                    stats = json.load(f)
                layer_name = json_file.stem.replace("_stats", "")
                debug_stats[layer_name] = stats
            except Exception as e:
                self._log(f"Warning: Could not load {json_file}: {e}")

        # Load scale stats
        scale_file = Path("./debug_scale/scale_stats.json")
        if scale_file.exists():
            try:
                with open(scale_file) as f:
                    debug_stats["_scale_stats"] = json.load(f)
            except Exception as e:
                self._log(f"Warning: Could not load scale stats: {e}")

        return debug_stats

    def analyze_sensitivity(
        self,
        debug_stats: dict[str, Any],
        current_exclusions: list[str],
    ) -> SensitivityAnalysisResult:
        """Analyze layer sensitivity from debug statistics."""
        self._log("Analyzing layer sensitivity...")

        analyzer = SensitivityAnalyzer(
            model=self.model,
            model_type=self.model_type,
            template=self.template,
            thresholds={
                "max_exclude_percent": self.config.max_exclude_percent,
            },
        )

        scale_stats = debug_stats.pop("_scale_stats", None)
        result = analyzer.analyze_from_calibration(debug_stats, scale_stats)

        # Print summary
        analyzer.print_summary(result)

        return result

    def update_exclusions(
        self,
        current_exclusions: list[str],
        sensitivity_result: SensitivityAnalysisResult,
    ) -> list[str]:
        """Update exclusion list based on sensitivity analysis."""
        new_exclusions = list(current_exclusions)

        # Add newly identified sensitive layers
        for layer in sensitivity_result.excluded_layers:
            if layer not in new_exclusions:
                new_exclusions.append(layer)

        # Convert to patterns
        patterns = convert_exclusions_to_patterns(new_exclusions)

        self._log(f"Updated exclusion patterns: {patterns}")
        return patterns

    def run_single_iteration(
        self,
        iteration: int,
        exclude_layers: list[str],
    ) -> IterationResult:
        """Run a single iteration of the quantization pipeline."""
        result = IterationResult(
            iteration=iteration,
            exclude_layers=exclude_layers,
        )

        try:
            # Create iteration-specific debug directory
            debug_dir = f"{self.config.debug_dir}/iteration_{iteration}"

            # Run calibration with debug
            quantized_model, debug_stats = self.run_calibration_with_debug(
                exclude_layers=exclude_layers,
                debug_dir=debug_dir,
            )

            # Evaluate quantized model
            if not self.config.skip_evaluation:
                result.quantized_ppl = self.evaluate_ppl(quantized_model)
                result.original_ppl = self.original_ppl
                result.ppl_degradation = result.quantized_ppl - self.original_ppl

            # Analyze sensitivity
            result.sensitivity_analysis = self.analyze_sensitivity(
                debug_stats,
                exclude_layers,
            )

            result.success = True
            result.timing = dict(self.timing)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._log(f"Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()

        return result

    def run(self) -> IterativeResult:
        """Run the complete iterative quantization pipeline."""
        total_start = time.time()
        final_result = IterativeResult(success=False)

        try:
            # Load model
            self.load_model()

            # Evaluate original model
            if not self.config.skip_evaluation:
                self.original_ppl = self.evaluate_ppl()
            else:
                self.original_ppl = None

            # Initialize exclusions
            current_exclusions = self.config.initial_exclusions or ["lm_head"]

            best_ppl_deg = float('inf')
            best_exclusions = current_exclusions
            best_iteration = 0

            # Iterative loop
            for iteration in range(1, self.config.max_iterations + 1):
                self._log(f"\n{'=' * 60}")
                self._log(f"ITERATION {iteration}/{self.config.max_iterations}")
                self._log(f"Current exclusions: {current_exclusions}")
                self._log(f"{'=' * 60}")

                # Run iteration
                iter_result = self.run_single_iteration(
                    iteration=iteration,
                    exclude_layers=current_exclusions,
                )
                final_result.iterations.append(iter_result)

                if not iter_result.success:
                    self._log(f"Iteration {iteration} failed, stopping.")
                    break

                # Check if this is the best so far
                if iter_result.ppl_degradation is not None:
                    if iter_result.ppl_degradation < best_ppl_deg:
                        best_ppl_deg = iter_result.ppl_degradation
                        best_exclusions = current_exclusions
                        best_iteration = iteration

                    # Check stopping criteria
                    if iter_result.ppl_degradation <= self.config.target_ppl_degradation:
                        self._log(f"Target PPL degradation reached: {iter_result.ppl_degradation:.4f}")
                        break

                # Update exclusions for next iteration
                if iter_result.sensitivity_analysis:
                    new_exclusions = self.update_exclusions(
                        current_exclusions,
                        iter_result.sensitivity_analysis,
                    )

                    # Check if exclusions changed
                    if set(new_exclusions) == set(current_exclusions):
                        self._log("No new exclusions identified, stopping.")
                        break

                    current_exclusions = new_exclusions

            # Final result
            final_result.success = True
            final_result.output_dir = self.config.output_dir
            final_result.best_iteration = best_iteration
            final_result.best_ppl_degradation = best_ppl_deg if best_ppl_deg != float('inf') else None
            final_result.final_exclude_layers = best_exclusions
            final_result.model_type = self.model_type
            final_result.total_timing = {
                "total": time.time() - total_start,
                **self.timing,
            }

            # Print final summary
            self._print_final_summary(final_result)

        except Exception as e:
            final_result.success = False
            final_result.error_message = str(e)
            self._log(f"Error in iterative quantization: {e}")
            import traceback
            traceback.print_exc()

        return final_result

    def _print_final_summary(self, result: IterativeResult) -> None:
        """Print final summary of iterative quantization."""
        print("\n" + "=" * 60)
        print("ITERATIVE QUANTIZATION SUMMARY")
        print("=" * 60)

        print(f"\nModel Type: {result.model_type}")
        print(f"Success: {result.success}")
        print(f"Total Iterations: {len(result.iterations)}")
        print(f"Best Iteration: {result.best_iteration}")

        if result.best_ppl_degradation is not None:
            print(f"\nBest PPL Degradation: {result.best_ppl_degradation:.4f}")

        print(f"\nFinal Exclusion Patterns ({len(result.final_exclude_layers)}):")
        for pattern in result.final_exclude_layers[:20]:
            print(f"  - {pattern}")
        if len(result.final_exclude_layers) > 20:
            print(f"  ... and {len(result.final_exclude_layers) - 20} more")

        print("\nIteration History:")
        for iter_result in result.iterations:
            status = "OK" if iter_result.success else "FAILED"
            ppl_str = f"PPL deg: {iter_result.ppl_degradation:.4f}" if iter_result.ppl_degradation else "N/A"
            print(f"  Iteration {iter_result.iteration}: {status}, {ppl_str}, "
                  f"{len(iter_result.exclude_layers)} exclusions")

        print(f"\nTotal Time: {result.total_timing.get('total', 0):.2f}s")
        print("=" * 60)


def run_iterative_quantization(
    model_path: str,
    output_dir: str,
    precision: str = "int4",
    max_iterations: int = 5,
    target_ppl_degradation: float = 2.0,
    **kwargs,
) -> IterativeResult:
    """
    Convenience function to run iterative quantization.

    Args:
        model_path: Path to the model
        output_dir: Output directory for quantized model
        precision: Target precision (int4, int8, fp8)
        max_iterations: Maximum number of iterations
        target_ppl_degradation: Target PPL degradation to stop early
        **kwargs: Additional configuration options

    Returns:
        IterativeResult with final quantization details
    """
    config = IterativeConfig(
        model_path=model_path,
        output_dir=output_dir,
        precision=precision,
        max_iterations=max_iterations,
        target_ppl_degradation=target_ppl_degradation,
        **kwargs,
    )

    quantizer = IterativeQuantizer(config)
    return quantizer.run()


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Iterative Quantization: Automatically find optimal layer exclusions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--precision", default="int4", choices=["int4", "int8", "fp8"])
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--target_ppl_degradation", type=float, default=2.0)
    parser.add_argument("--max_exclude_percent", type=float, default=20.0)
    parser.add_argument("--calibration_data", default="pileval")
    parser.add_argument("--num_calib_samples", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory_efficient", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--debug_dir", default="./debug_output")

    args = parser.parse_args()

    result = run_iterative_quantization(
        model_path=args.model_path,
        output_dir=args.output_dir,
        precision=args.precision,
        max_iterations=args.max_iterations,
        target_ppl_degradation=args.target_ppl_degradation,
        max_exclude_percent=args.max_exclude_percent,
        calibration_data=args.calibration_data,
        num_calib_samples=args.num_calib_samples,
        device=args.device,
        memory_efficient=args.memory_efficient,
        skip_evaluation=args.skip_evaluation,
        debug_dir=args.debug_dir,
    )

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = Path(args.output_dir) / "iterative_result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {result_path}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Sequential Sensitivity Analyzer for Layer-wise Quantization.

Performs memory-efficient sensitivity analysis by processing layers
sequentially, capturing cascading quantization effects.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils import clear_gpu_memory
from .activation_cache import ActivationCache, CacheLocation
from .scorer import SensitivityMetric, SensitivityScorer

if TYPE_CHECKING:
    from ..config import UnifiedConfig


@dataclass
class AnalysisResult:
    """Result of sensitivity analysis."""
    success: bool
    sensitive_layers: dict[str, float]  # layer_name -> score
    excluded_layers: list[str]  # layers to exclude from quantization
    timing: dict[str, float]
    error_message: str | None = None


class SequentialSensitivityAnalyzer:
    """
    Memory-efficient sequential sensitivity analyzer.

    Analyzes layer sensitivity to quantization by:
    1. Running baseline forward pass (FP16), caching activations on GPU
    2. For each layer, comparing quantized output vs baseline

    This captures cascading quantization effects that independent
    layer analysis misses.

    Usage:
        config = UnifiedConfig(
            model_path="meta-llama/Llama-3-8B",
            sensitivity_threshold=0.02,
        )

        analyzer = SequentialSensitivityAnalyzer(config)
        result = analyzer.analyze()

        print(f"Sensitive layers: {result.sensitive_layers}")
        print(f"Exclude: {result.excluded_layers}")
    """

    def __init__(
        self,
        config: UnifiedConfig,
        metric: SensitivityMetric = SensitivityMetric.RELATIVE_NORM,
        cache_on_gpu: bool = True,
    ):
        """
        Initialize the analyzer.

        Args:
            config: UnifiedConfig with model and analysis settings
            metric: Sensitivity metric to use
            cache_on_gpu: Store activations on GPU by default
        """
        self.config = config
        self.metric = metric
        self.cache_on_gpu = cache_on_gpu

        # Components
        self.cache = ActivationCache(
            device=config.device,
            gpu_memory_threshold=0.7,  # Use 70% of GPU for activations
            enable_cpu_spillover=True,
        )
        self.scorer = SensitivityScorer(metric=metric)

        # State
        self.hf_config = None
        self.tokenizer = None
        self.model = None
        self.layer_prefix = "model.layers"
        self.num_layers = 0

    def _log(self, message: str) -> None:
        """Print log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _setup(self) -> None:
        """Load model configuration and tokenizer."""
        from transformers import AutoConfig, AutoTokenizer

        self._log("Setting up sensitivity analysis...")

        # Load config
        self.hf_config = AutoConfig.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Get layer info
        self.num_layers = getattr(self.hf_config, "num_hidden_layers", 0)
        self.layer_prefix = "model.layers"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._log(f"Model has {self.num_layers} layers")

    def _get_calibration_input(self) -> torch.Tensor:
        """Get calibration input tensor."""
        from ..utils import get_calib_dataloader

        calib_loader = get_calib_dataloader(
            dataset_name_or_path=self.config.calibration_data,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            num_calib_data=self.config.num_calib_samples,
            seqlen=self.config.seq_len,
            device=self.config.device,
        )

        # Get first batch
        for batch in calib_loader:
            if isinstance(batch, dict):
                return batch.get("input_ids", batch.get("input", None))
            return batch

        raise ValueError("No calibration data available")

    def _load_model_layer_by_layer(self) -> list[nn.Module]:
        """
        Load model layers one by one to minimize memory usage.

        Returns:
            List of layer modules
        """
        from transformers import AutoModelForCausalLM

        self._log("Loading model layers...")

        # Load full model (TODO: implement true layer-by-layer loading)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
        )

        self.model = model

        # Extract layers
        layers = []
        for i in range(self.num_layers):
            layer_name = f"{self.layer_prefix}.{i}"
            layer = dict(model.named_modules()).get(layer_name)
            if layer is not None:
                layers.append(layer)

        self._log(f"Extracted {len(layers)} layers")
        return layers

    def _run_baseline_pass(self, layers: list[nn.Module], input_tensor: torch.Tensor) -> None:
        """
        Run baseline forward pass, caching activations on GPU.

        Args:
            layers: List of model layers
            input_tensor: Input tensor for the model
        """
        self._log("Running baseline pass (FP16)...")

        # Store initial input
        current_activation = input_tensor.to(self.config.device)
        self.cache.store(layer_idx=0, activation=current_activation, is_input=True)

        # Process each layer
        for idx, layer in enumerate(tqdm(layers, desc="Baseline pass")):
            # Move layer to GPU
            layer = layer.to(self.config.device)

            # Forward pass
            with torch.no_grad():
                # Handle different layer output formats
                output = layer(current_activation)

                if isinstance(output, tuple):
                    layer_output = output[0]
                else:
                    layer_output = output

            # Cache output (which is input to next layer)
            current_activation = layer_output

            # Store on GPU by default
            self.cache.store(
                layer_idx=idx + 1,
                activation=current_activation,
                is_input=True,
                force_location=CacheLocation.GPU if self.cache_on_gpu else None,
            )

            # Move layer back to CPU
            layer = layer.cpu()
            clear_gpu_memory()

        self._log(f"Baseline pass complete. Cache: {self.cache.get_memory_summary()}")

    def _quantize_single_layer(
        self,
        layer: nn.Module,
        layer_name: str,
    ) -> nn.Module:
        """
        Quantize a single layer for sensitivity testing.

        Args:
            layer: The layer module to quantize
            layer_name: Name of the layer (for logging)

        Returns:
            Quantized layer
        """
        from quark.torch import ModelQuantizer
        from quark.torch.quantization.config.config import Int4PerGroupSpec, QConfig, QLayerConfig

        # Create quantization config for this layer only
        quant_config = QConfig(
            global_quant_config=QLayerConfig(
                weight=Int4PerGroupSpec(group_size=128).to_quantization_spec()
            ),
        )

        # Quantize
        quantizer = ModelQuantizer(quant_config)

        from torch.utils.data import DataLoader, TensorDataset
        dummy = torch.zeros(1, 1, device=self.config.device)
        dummy_loader = DataLoader(TensorDataset(dummy), batch_size=1)

        layer = layer.to(self.config.device)
        quantized_layer = quantizer.quantize_model(layer, dummy_loader)
        quantized_layer = quantizer.freeze(quantized_layer)

        return quantized_layer

    def _run_sensitivity_pass(self, layers: list[nn.Module]) -> None:
        """
        Run sensitivity analysis pass.

        For each layer:
        1. Quantize that layer
        2. Run forward using cached input
        3. Compare output to cached baseline
        """
        self._log("Running sensitivity analysis pass...")

        for idx, layer in enumerate(tqdm(layers, desc="Sensitivity pass")):
            layer_name = f"{self.layer_prefix}.{idx}"

            # Get cached input for this layer
            cached_input = self.cache.get(
                layer_idx=idx,
                is_input=True,
                target_device=self.config.device,
            )

            if cached_input is None:
                self._log(f"  Warning: No cached input for layer {idx}")
                continue

            # Clone layer for quantization
            import copy
            layer_copy = copy.deepcopy(layer)

            # Quantize the layer
            try:
                quantized_layer = self._quantize_single_layer(layer_copy, layer_name)
            except Exception as e:
                self._log(f"  Error quantizing layer {idx}: {e}")
                continue

            # Get baseline output
            baseline_output = self.cache.get(
                layer_idx=idx + 1,
                is_input=True,
                target_device=self.config.device,
            )

            if baseline_output is None:
                # Run baseline forward to get output
                layer_fp16 = layer.to(self.config.device)
                with torch.no_grad():
                    output = layer_fp16(cached_input)
                    if isinstance(output, tuple):
                        baseline_output = output[0]
                    else:
                        baseline_output = output
                layer_fp16 = layer_fp16.cpu()

            # Run quantized forward
            with torch.no_grad():
                output = quantized_layer(cached_input)
                if isinstance(output, tuple):
                    quantized_output = output[0]
                else:
                    quantized_output = output

            # Record sensitivity score
            self.scorer.record_layer_score(
                layer_name=layer_name,
                layer_idx=idx,
                baseline_output=baseline_output,
                quantized_output=quantized_output,
            )

            # Cleanup
            del layer_copy, quantized_layer
            clear_gpu_memory()

        self._log("Sensitivity pass complete")

    def analyze(self) -> AnalysisResult:
        """
        Run sequential sensitivity analysis.

        Returns:
            AnalysisResult with sensitivity scores and excluded layers
        """
        total_start = time.time()
        result = AnalysisResult(
            success=False,
            sensitive_layers={},
            excluded_layers=[],
            timing={},
        )

        try:
            # Setup
            setup_start = time.time()
            self._setup()
            result.timing["setup"] = time.time() - setup_start

            # Load layers
            load_start = time.time()
            layers = self._load_model_layer_by_layer()
            result.timing["load_model"] = time.time() - load_start

            # Get calibration input
            input_tensor = self._get_calibration_input()

            # Baseline pass
            baseline_start = time.time()
            self._run_baseline_pass(layers, input_tensor)
            result.timing["baseline_pass"] = time.time() - baseline_start

            # Sensitivity pass
            sensitivity_start = time.time()
            self._run_sensitivity_pass(layers)
            result.timing["sensitivity_pass"] = time.time() - sensitivity_start

            # Aggregate results
            scores = self.scorer.get_aggregated_scores()
            result.sensitive_layers = {s.layer_name: s.score for s in scores}

            # Determine excluded layers based on threshold
            threshold = self.config.sensitivity_threshold
            if threshold > 0:
                result.excluded_layers = self.scorer.get_layers_above_threshold(threshold)
                self._log(f"Layers above threshold {threshold}: {result.excluded_layers}")

            result.success = True
            result.timing["total"] = time.time() - total_start

            self._print_summary(result)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._log(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            self.cache.clear()
            if self.model is not None:
                del self.model
            clear_gpu_memory()
            gc.collect()

        return result

    def _print_summary(self, result: AnalysisResult) -> None:
        """Print analysis summary."""
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Layers analyzed: {len(result.sensitive_layers)}")

        if result.sensitive_layers:
            print("\nTop 5 most sensitive layers:")
            sorted_layers = sorted(
                result.sensitive_layers.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for name, score in sorted_layers[:5]:
                print(f"  {name}: {score:.6f}")

        if result.excluded_layers:
            print(f"\nLayers to exclude (threshold={self.config.sensitivity_threshold}):")
            for name in result.excluded_layers:
                print(f"  {name}")

        if result.timing:
            print("\nTiming:")
            for stage, duration in result.timing.items():
                print(f"  {stage}: {duration:.2f}s")

        print("=" * 60)

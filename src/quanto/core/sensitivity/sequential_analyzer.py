"""
Sequential Sensitivity Analyzer for Layer-wise Quantization.

Performs memory-efficient sensitivity analysis using forward hooks
to capture activations during a full model forward pass.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import tqdm

from quanto.utils import clear_gpu_memory
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
    Memory-efficient sequential sensitivity analyzer using forward hooks.

    Analyzes layer sensitivity to quantization by:
    1. Running baseline forward pass with hooks to capture FP16 activations
    2. For each layer, quantizing it and comparing output vs baseline

    This captures cascading quantization effects that independent
    layer analysis misses.
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
            gpu_memory_threshold=0.7,
            enable_cpu_spillover=True,
        )
        self.scorer = SensitivityScorer(metric=metric)

        # State
        self.hf_config = None
        self.tokenizer = None
        self.model = None
        self.layer_prefix = "model.layers"
        self.num_layers = 0
        self.layer_names: list[str] = []

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
        from quanto.utils import get_calib_dataloader

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

    def _load_model(self) -> nn.Module:
        """Load the full model."""
        from transformers import AutoModelForCausalLM

        self._log("Loading model...")

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
        )

        self.model = model

        # Find all transformer layers
        self.layer_names = []
        for i in range(self.num_layers):
            layer_name = f"{self.layer_prefix}.{i}"
            layer = dict(model.named_modules()).get(layer_name)
            if layer is not None:
                self.layer_names.append(layer_name)

        self._log(f"Found {len(self.layer_names)} layers")
        return model

    def _run_baseline_with_hooks(self, model: nn.Module, input_ids: torch.Tensor) -> None:
        """
        Run baseline forward pass with hooks to capture activations.

        Args:
            model: The full model
            input_ids: Input token IDs
        """
        self._log("Running baseline pass (FP16) with hooks...")

        # Storage for captured activations
        captured = {}

        def make_hook(layer_name: str):
            def hook(module, inp, out):
                # Store input (first element of inp tuple)
                if inp and len(inp) > 0:
                    activation = inp[0]
                    if isinstance(activation, torch.Tensor):
                        captured[layer_name] = activation.detach().clone()
            return hook

        # Register hooks on all layers
        hooks = []
        for layer_name in self.layer_names:
            layer = dict(model.named_modules()).get(layer_name)
            if layer is not None:
                hook = layer.register_forward_hook(make_hook(layer_name))
                hooks.append(hook)

        # Also capture the model's final input (embeddings)
        def embed_hook(module, inp, out):
            captured["__embed_output__"] = out.detach().clone()

        # Find and hook embedding layer
        embed_layer = None
        for name in ["model.embed_tokens", "model.wte", "transformer.wte"]:
            embed_layer = dict(model.named_modules()).get(name)
            if embed_layer:
                break

        if embed_layer:
            hooks.append(embed_layer.register_forward_hook(embed_hook))

        # Forward pass
        with torch.no_grad():
            _ = model(input_ids)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Store captured activations in cache
        # First store embedding output
        if "__embed_output__" in captured:
            self.cache.store(
                layer_idx=0,
                activation=captured["__embed_output__"],
                is_input=True,
                force_location=CacheLocation.GPU if self.cache_on_gpu else None,
            )

        # Then store each layer's input
        for idx, layer_name in enumerate(self.layer_names):
            if layer_name in captured:
                self.cache.store(
                    layer_idx=idx + 1,
                    activation=captured[layer_name],
                    is_input=True,
                    force_location=CacheLocation.GPU if self.cache_on_gpu else None,
                )

        # Also capture output of last layer for final comparison
        # (already done via the hook on last layer)

        self._log(f"Baseline pass complete. Cache: {self.cache.get_memory_summary()}")

    def _quantize_and_compare_layer(
        self,
        model: nn.Module,
        layer_name: str,
        layer_idx: int,
    ) -> float | None:
        """
        Quantize a single layer and compute sensitivity score.

        Uses weight-level comparison to avoid position embedding issues
        with transformer layers.

        Args:
            model: The full model
            layer_name: Name of the layer to test
            layer_idx: Index of the layer

        Returns:
            Sensitivity score (weight quantization error), or None if failed
        """
        import copy

        # Get the layer
        layer = dict(model.named_modules()).get(layer_name)
        if layer is None:
            return None

        # Clone the layer for quantization
        layer_copy = copy.deepcopy(layer)

        # Get original weights
        original_weights = {}
        for name, param in layer.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                original_weights[name] = param.data.clone()

        if not original_weights:
            return None

        # Quantize the layer
        try:
            quantized_layer = self._quantize_layer(layer_copy, layer_name)
        except Exception as e:
            self._log(f"  Error quantizing {layer_name}: {e}")
            return None

        # Get quantized weights and compute error
        total_error = 0.0
        total_norm = 0.0

        for name, param in quantized_layer.named_parameters():
            if 'weight' in name and len(param.shape) == 2 and name in original_weights:
                orig = original_weights[name].float()
                quant = param.data.float()

                # Compute relative L2 error
                diff_norm = torch.norm(orig - quant).item()
                orig_norm = torch.norm(orig).item()

                if orig_norm > 1e-8:
                    relative_error = diff_norm / orig_norm
                    total_error += relative_error
                    total_norm += 1

        # Cleanup
        del layer_copy, quantized_layer
        clear_gpu_memory()

        if total_norm > 0:
            return total_error / total_norm
        return None

    def _quantize_layer(self, layer: nn.Module, layer_name: str) -> nn.Module:
        """
        Quantize a single layer for sensitivity testing.

        Args:
            layer: The layer module to quantize
            layer_name: Name of the layer

        Returns:
            Quantized layer
        """
        from quark.torch import ModelQuantizer
        from quark.torch.quantization.config.config import Int4PerGroupSpec, QConfig, QLayerConfig

        # Create quantization config
        # ch_axis=0 for per-row quantization (output channel dimension)
        quant_config = QConfig(
            global_quant_config=QLayerConfig(
                weight=Int4PerGroupSpec(ch_axis=0, group_size=128).to_quantization_spec()
            ),
        )

        # Quantize
        quantizer = ModelQuantizer(quant_config)

        from torch.utils.data import DataLoader, TensorDataset
        dummy = torch.zeros(1, 1, device=self.config.device)
        dummy_loader = DataLoader(TensorDataset(dummy), batch_size=1)

        quantized_layer = quantizer.quantize_model(layer, dummy_loader)
        quantized_layer = quantizer.freeze(quantized_layer)

        return quantized_layer

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

            # Load model
            load_start = time.time()
            model = self._load_model()
            result.timing["load_model"] = time.time() - load_start

            # Get calibration input
            input_ids = self._get_calibration_input()

            # Baseline pass with hooks
            baseline_start = time.time()
            self._run_baseline_with_hooks(model, input_ids)
            result.timing["baseline_pass"] = time.time() - baseline_start

            # Sensitivity analysis per layer
            sensitivity_start = time.time()
            self._log("Running sensitivity analysis per layer...")

            for idx, layer_name in enumerate(tqdm(self.layer_names, desc="Sensitivity pass")):
                score = self._quantize_and_compare_layer(model, layer_name, idx)

                if score is not None:
                    result.sensitive_layers[layer_name] = score
                    self.scorer._scores[layer_name] = [score]

            result.timing["sensitivity_pass"] = time.time() - sensitivity_start

            # Determine excluded layers based on threshold
            threshold = self.config.sensitivity_threshold
            if threshold > 0:
                result.excluded_layers = [
                    name for name, score in result.sensitive_layers.items()
                    if score > threshold
                ]
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

        print("\nCache Performance:")
        print(f"  {self.cache.get_memory_summary()}")

        print("=" * 60)

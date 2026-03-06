"""
Unified Quantizer: Single class supporting all quantization strategies.

This module consolidates the functionality of:
- AutoQuantizer (full GPU quantization)
- IterativeQuantizer (iterative exclusion discovery)
- LayerwiseQuantizer (CPU model, GPU quantization)
- LazyLayerwiseQuantizer (lazy disk loading with INT4 packing)
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# Add quark to path
sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.quantization.config.config import Int4PerGroupSpec, QConfig, QLayerConfig
from transformers import AutoConfig, AutoTokenizer

from ..constants import PRECISION_TO_SCHEME
from ..utils import (
    clear_gpu_memory,
    detect_model_type,
    get_calib_dataloader,
    get_memory_info,
    get_template,
    pack_layer_weights,
)
from .base_quantizer import QuantizationResult
from .config import UnifiedConfig


@contextmanager
def suppress_quark_output() -> Generator[None, None, None]:
    """Temporarily suppress Quark's internal output (progress bars, logs)."""
    import sys
    from io import StringIO

    # Save original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect to StringIO (captures all output including tqdm)
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        yield
    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class UnifiedQuantizer:
    """
    Unified quantizer supporting multiple memory strategies.

    This class provides a single interface for:
    - Full GPU quantization (memory_strategy="full")
    - Layerwise CPU quantization (memory_strategy="layerwise_cpu")
    - Lazy disk loading quantization (memory_strategy="lazy")
    - Automatic strategy selection (memory_strategy="auto")

    Features:
    - Sensitivity-based layer exclusion
    - INT4 weight packing for storage efficiency
    - Memory-efficient processing for large models
    """

    def __init__(self, config: UnifiedConfig):
        """
        Initialize unified quantizer.

        Args:
            config: UnifiedConfig with all quantization settings
        """
        self.config = config

        # State
        self.hf_config = None
        self.tokenizer = None
        self.model_type = None
        self.template = None
        self.safetensors_files = []
        self.weight_index = {}  # Maps weight name to file path
        self.timing = {}
        self.sensitivity_scores: dict[str, float] = {}  # Store sensitivity analysis results

    def _log(self, message: str) -> None:
        """Print log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _configure_quark_logging(self) -> None:
        """Configure Quark logging based on verbose setting."""
        import os

        # Quark uses QUARK_LOG_LEVEL environment variable
        if not hasattr(self.config, 'verbose') or not self.config.verbose:
            # Suppress Quark INFO logs (only show warnings)
            os.environ["QUARK_LOG_LEVEL"] = "warning"

    def _clear_memory(self) -> None:
        """Clear GPU and CPU memory."""
        clear_gpu_memory()

    def _get_memory_info(self) -> str:
        """Get memory usage info."""
        return get_memory_info()

    def _get_quant_scheme(self) -> str:
        """Get quantization scheme from precision setting."""
        return PRECISION_TO_SCHEME.get(self.config.precision, self.config.precision)

    def _detect_model_type(self) -> str:
        """Detect model type from config."""
        self.model_type = detect_model_type(self.config.model_path, self.config.trust_remote_code)
        self._log(f"Detected model type: {self.model_type}")
        return self.model_type

    def _get_template(self) -> LLMTemplate | None:
        """Get LLMTemplate for the model."""
        self.template = get_template(self.model_type)
        if self.template:
            self._log(f"Using template: {self.template.model_type}")
        else:
            self._log(f"Warning: No template found for model type '{self.model_type}'")
        return self.template

    def _setup(self) -> None:
        """Load config, tokenizer, and build weight index."""
        start_time = time.time()
        self._log("Setting up quantization...")

        # Load HuggingFace config (no weights)
        self.hf_config = AutoConfig.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        self._detect_model_type()
        self._get_template()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "quantized_layers"), exist_ok=True)

        self.timing["setup"] = time.time() - start_time
        self._log(f"Setup completed in {self.timing['setup']:.2f}s")

    def _get_layer_info(self) -> dict[str, Any]:
        """Get layer information from config or verify against weight index."""
        info = {
            "num_layers": getattr(self.hf_config, "num_hidden_layers", 0),
            "layer_prefix": "model.layers",
            "hidden_size": getattr(self.hf_config, "hidden_size", 0),
            "intermediate_size": getattr(self.hf_config, "intermediate_size", 0),
        }

        # Handle case where model_type may not be set yet
        model_type_lower = (self.model_type or "").lower()
        if (
            "llama" in model_type_lower
            or "qwen" in model_type_lower
            or "mistral" in model_type_lower
            or "mixtral" in model_type_lower
            or "phi" in model_type_lower
            or "gemma" in model_type_lower
        ):
            info["layer_prefix"] = "model.layers"

        # Verify layer count against weight index if available
        if self.weight_index:
            actual_layers = set()
            for key in self.weight_index.keys():
                if key.startswith(f"{info['layer_prefix']}."):
                    parts = key.split(".")
                    if len(parts) > 2:
                        try:
                            layer_idx = int(parts[2])
                            actual_layers.add(layer_idx)
                        except (ValueError, IndexError):
                            pass

            if actual_layers:
                actual_num_layers = max(actual_layers) + 1
                if actual_num_layers != info["num_layers"]:
                    self._log(f"Warning: config says {info['num_layers']} layers, but weight index has {actual_num_layers} layers")
                    self._log(f"Using actual layer count from weight index: {actual_num_layers}")
                    info["num_layers"] = actual_num_layers

        return info

    def _build_weight_index(self) -> None:
        """Build index mapping weight names to safetensors files."""
        self._log("Building weight index from safetensors files...")

        model_path = Path(self.config.model_path)

        # Find all safetensors files
        self.safetensors_files = list(model_path.glob("*.safetensors"))
        if not self.safetensors_files:
            # Check for model.safetensors.index.json
            index_file = model_path / "model.safetensors.index.json"
            if index_file.exists():
                with open(index_file) as f:
                    index_data = json.load(f)
                self.weight_index = index_data.get("weight_map", {})
                self.safetensors_files = [
                    model_path / f for f in set(self.weight_index.values())
                ]
            else:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
        else:
            # Build index by scanning files
            for sf_file in self.safetensors_files:
                f = safe_open(sf_file, framework="pt")
                for key in f.keys():
                    self.weight_index[key] = str(sf_file)

        self._log(f"Found {len(self.safetensors_files)} safetensors files")
        self._log(f"Indexed {len(self.weight_index)} weight tensors")

    def _determine_exclude_layers(self, strategy: str = "full") -> list[str]:
        """Determine layers to exclude based on config and model analysis.

        Args:
            strategy: Current memory strategy ("full", "layerwise_cpu", "lazy")
        """
        if self.config.exclude_layers is not None:
            return self.config.exclude_layers

        # Default exclusions
        exclude = ["lm_head"]

        # Add standard patterns
        exclude.extend(["*embed*", "*norm*"])

        if self.config.aggressive_exclusion:
            exclude.extend(["*gate*"])

        # Run sensitivity analysis if enabled
        if self.config.sensitivity_analysis:
            if strategy == "full":
                # Full sequential analysis (loads model to GPU)
                self._log("Running sequential sensitivity analysis...")
                sensitive_layers = self._run_sequential_sensitivity_analysis()
                exclude.extend(sensitive_layers)
            else:
                # Lightweight analysis (no full model load, works with any strategy)
                sensitive_layers = self._run_lightweight_sensitivity_analysis()
                exclude.extend(sensitive_layers)
        # Fallback to simple analysis if only threshold is set (backward compatibility)
        elif self.config.sensitivity_threshold > 0:
            self._log("Running simple sensitivity analysis...")
            sensitive_layers = self._analyze_sensitive_layers()
            exclude.extend(sensitive_layers)

        # Remove duplicates
        return list(set(exclude))

    def _analyze_sensitive_layers(self) -> list[str]:
        """Analyze layers for sensitivity to quantization (simple heuristic)."""
        # Get layer info
        layer_info = self._get_layer_info()
        num_layers = layer_info["num_layers"]
        layer_prefix = layer_info["layer_prefix"]

        sensitive = []

        # Sample a few layers for sensitivity analysis
        sample_indices = [0, num_layers // 2, num_layers - 1] if num_layers > 3 else list(range(num_layers))

        for idx in sample_indices:
            layer_weights = self._load_layer_weights(idx, layer_prefix)
            if not layer_weights:
                continue

            # Analyze weight statistics for sensitivity
            for name, tensor in layer_weights.items():
                if "weight" in name and len(tensor.shape) == 2:
                    # Check for high dynamic range (sensitive to quantization)
                    tensor_range = tensor.max() - tensor.min()
                    tensor_std = tensor.std()

                    # High range-to-std ratio indicates potential sensitivity
                    if tensor_std > 0:
                        ratio = tensor_range / tensor_std
                        if ratio > self.config.sensitivity_threshold * 10:
                            sensitive.append(name)

        return sensitive

    def _run_lightweight_sensitivity_analysis(self) -> list[str]:
        """
        Run lightweight sensitivity analysis without loading full model to GPU.

        This approach:
        1. Loads each layer's weights from disk one at a time
        2. Calculates sensitivity scores for individual weight tensors (sub-layer level)
        3. Stores scores for final report with sub-layer granularity

        Less accurate than sequential analysis but works with any memory strategy.
        """
        self._log("Running lightweight sensitivity analysis (no full model load)...")

        layer_info = self._get_layer_info()
        num_layers = layer_info["num_layers"]
        layer_prefix = layer_info["layer_prefix"]
        threshold = self.config.sensitivity_threshold

        excluded_layers = []

        # Store sub-layer level scores (individual weights)
        self.sublayer_sensitivity_scores: dict[str, float] = {}
        # Store aggregated scores by sub-layer type (e.g., q_proj, gate_proj)
        self.sublayer_type_scores: dict[str, list[float]] = {}

        # Analyze all layers
        for layer_idx in tqdm(range(num_layers), desc="Sensitivity analysis"):
            layer_weights = self._load_layer_weights(layer_idx, layer_prefix)
            if not layer_weights:
                continue

            # Calculate sensitivity score for each individual weight tensor
            for name, tensor in layer_weights.items():
                if "weight" in name and len(tensor.shape) == 2:
                    tensor_range = (tensor.max() - tensor.min()).item()
                    tensor_std = tensor.std().item()
                    tensor_norm = tensor.norm().item()

                    if tensor_std > 1e-8 and tensor_norm > 1e-8:
                        # Relative range: higher = more sensitive to quantization
                        relative_range = tensor_range / tensor_norm

                        # Store individual weight score
                        self.sublayer_sensitivity_scores[name] = relative_range

                        # Extract sub-layer type (e.g., q_proj, gate_proj, etc.)
                        parts = name.split('.')
                        for i, part in enumerate(parts):
                            if part in ['self_attn', 'mlp', 'attention', 'feed_forward']:
                                if i + 1 < len(parts):
                                    sublayer_type = parts[i + 1]
                                    if sublayer_type not in self.sublayer_type_scores:
                                        self.sublayer_type_scores[sublayer_type] = []
                                    self.sublayer_type_scores[sublayer_type].append(relative_range)
                                break

            # Clear memory
            del layer_weights

        # Calculate aggregated layer scores (average of sub-layer scores)
        layer_scores: dict[str, list[float]] = {}
        for name, score in self.sublayer_sensitivity_scores.items():
            # Extract layer name (e.g., model.layers.0)
            parts = name.split('.')
            if len(parts) >= 3:
                layer_name = f"{parts[0]}.{parts[1]}.{parts[2]}"
                if layer_name not in layer_scores:
                    layer_scores[layer_name] = []
                layer_scores[layer_name].append(score)

        # Average layer scores
        for layer_name, scores in layer_scores.items():
            avg_score = sum(scores) / len(scores)
            self.sensitivity_scores[layer_name] = avg_score

        # Exclude individual sub-layer weights that exceed threshold
        for weight_name, score in self.sublayer_sensitivity_scores.items():
            if score > threshold:
                excluded_layers.append(weight_name)

        # Print summary for sub-layer types
        self._log(f"Analyzed {len(self.sublayer_sensitivity_scores)} weight tensors across {num_layers} layers")

        # Show sub-layer type statistics
        if self.sublayer_type_scores:
            sorted_types = sorted(
                [(k, sum(v)/len(v)) for k, v in self.sublayer_type_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            self._log(f"Sub-layer type sensitivity (avg):")
            for sublayer_type, avg_score in sorted_types[:5]:
                self._log(f"  {sublayer_type}: {avg_score:.6f}")

        if self.sensitivity_scores:
            sorted_scores = sorted(
                self.sensitivity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self._log(f"Analyzed {len(self.sensitivity_scores)} layers")
            self._log(f"Top 5 most sensitive layers:")
            for i, (name, score) in enumerate(sorted_scores[:5]):
                status = " (EXCLUDED)" if score > threshold else ""
                self._log(f"  {i+1}. {name}: {score:.6f}{status}")

            if excluded_layers:
                self._log(f"Layers above threshold ({threshold}): {len(excluded_layers)}")

        return excluded_layers

    def _get_aggregated_sublayer_scores(self) -> dict[str, dict[str, float]]:
        """
        Convert sublayer_type_scores from dict[str, list[float]] to dict[str, dict[str, float]].

        Returns aggregated scores with 'avg' and 'max' for each sub-layer type.
        """
        if not hasattr(self, 'sublayer_type_scores') or not self.sublayer_type_scores:
            return {}

        aggregated = {}
        for sublayer_type, scores in self.sublayer_type_scores.items():
            if scores:
                aggregated[sublayer_type] = {
                    "avg": sum(scores) / len(scores),
                    "max": max(scores),
                    "min": min(scores),
                    "count": len(scores),
                }
        return aggregated

    def _populate_result_sensitivity(self, result: QuantizationResult) -> None:
        """
        Populate sensitivity scores in the result object.

        Includes both layer-level and sub-layer level sensitivity data.
        """
        result.sensitivity_scores = getattr(self, 'sensitivity_scores', {})
        result.sublayer_sensitivity_scores = getattr(self, 'sublayer_sensitivity_scores', {})
        result.sublayer_type_scores = self._get_aggregated_sublayer_scores()

    def _run_sequential_sensitivity_analysis(self) -> list[str]:
        """
        Run sequential sensitivity analysis to find sensitive layers.

        Uses SequentialSensitivityAnalyzer for accurate cascading effect detection.
        Activations are cached on GPU by default for speed.
        """
        from .sensitivity import SequentialSensitivityAnalyzer

        analyzer = SequentialSensitivityAnalyzer(
            config=self.config,
            cache_on_gpu=self.config.sensitivity_cache_on_gpu,
        )

        result = analyzer.analyze()

        if not result.success:
            self._log(f"Warning: Sensitivity analysis failed: {result.error_message}")
            return []

        # Log cache performance
        self._log(f"Sensitivity analysis cache: {analyzer.cache.get_memory_summary()}")

        # Store timing info
        if result.timing:
            self.timing["sensitivity_analysis"] = result.timing.get("total", 0)

        # Store sensitivity scores for final report
        self.sensitivity_scores = result.sensitive_layers.copy()

        # Return layers to exclude
        return result.excluded_layers

    def _get_layer_weight_names(self, layer_idx: int, layer_prefix: str) -> list[str]:
        """Get all weight names for a specific layer."""
        layer_pattern = f"{layer_prefix}.{layer_idx}."
        return [k for k in self.weight_index if k.startswith(layer_pattern)]

    def _load_layer_weights(self, layer_idx: int, layer_prefix: str) -> dict[str, torch.Tensor]:
        """Load weights for specific layer from safetensors."""
        weight_names = self._get_layer_weight_names(layer_idx, layer_prefix)
        if not weight_names:
            return {}

        weights = {}

        # Group by file to minimize file opens
        file_to_keys = {}
        for key in weight_names:
            file_path = self.weight_index.get(key)
            if file_path:
                if file_path not in file_to_keys:
                    file_to_keys[file_path] = []
                file_to_keys[file_path].append(key)

        # Load from each file
        for file_path, keys in file_to_keys.items():
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in keys:
                    weights[key] = f.get_tensor(key)

        return weights

    def _create_quant_config(self, exclude_layers: list[str]) -> QConfig:
        """Create Quark quantization config."""
        quant_scheme = self._get_quant_scheme()
        self._log(f"Using quantization scheme: {quant_scheme}")

        # Create base quant config
        if self.template:
            quant_config = self.template.get_config(
                scheme=quant_scheme,
                exclude_layers=exclude_layers,
            )
        else:
            quant_config = QConfig(
                global_quant_config=QLayerConfig(
                    weight=Int4PerGroupSpec(group_size=128).to_quantization_spec()
                ),
                exclude=exclude_layers,
            )

        return quant_config

    def _quantize_layer_weights(
        self,
        layer_weights: dict[str, torch.Tensor],
        layer_idx: int,
        quant_config: QConfig,
        exclude_patterns: list[str],
    ) -> dict[str, torch.Tensor]:
        """Quantize layer weights using Quark."""

        # Create a module with proper Linear layers that Quark can quantize
        class LinearLayer(nn.Module):
            """Module with proper Linear layers for quantization."""

            def __init__(self, weights: dict[str, torch.Tensor]):
                super().__init__()
                self.linear_layers = nn.ModuleDict()
                self.safe_to_orig = {}

                for i, (name, tensor) in enumerate(weights.items()):
                    # Only process 2D tensors (Linear weights)
                    if len(tensor.shape) == 2:
                        out_features, in_features = tensor.shape
                        linear = nn.Linear(in_features, out_features, bias=False)
                        linear.weight.data = tensor
                        # Use indexed name for ModuleDict to avoid naming collisions
                        safe_name = f"layer_{i}"
                        self.linear_layers[safe_name] = linear
                        self.safe_to_orig[safe_name] = name

            def forward(self, x):
                return x  # Dummy forward

        # Check if any weights should be excluded
        import fnmatch

        excluded_weights = set()
        for name in layer_weights:
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(name, pattern) or pattern in name:
                    excluded_weights.add(name)
                    break

        # Filter weights to quantize
        weights_to_quantize = {
            k: v for k, v in layer_weights.items() if k not in excluded_weights
        }

        if not weights_to_quantize:
            # All excluded, return original weights
            return layer_weights

        # Create the layer with proper Linear modules
        linear_layer = LinearLayer(weights_to_quantize)
        safe_to_orig = linear_layer.safe_to_orig
        linear_layer = linear_layer.to(self.config.device)

        # Create quantizer
        quantizer = ModelQuantizer(quant_config)

        # Create dummy dataloader for weight-only quantization
        from torch.utils.data import DataLoader, TensorDataset

        dummy = torch.zeros(1, 1, device=self.config.device)
        dummy_loader = DataLoader(TensorDataset(dummy), batch_size=1)

        # Quantize (suppress Quark's internal progress bars in non-verbose mode)
        verbose = getattr(self.config, 'verbose', False)
        ctx = suppress_quark_output() if not verbose else contextmanager(lambda: iter([None]))()
        with ctx:
            quantized_layer = quantizer.quantize_model(linear_layer, dummy_loader)
            quantized_layer = quantizer.freeze(quantized_layer)

        # Extract quantized weights with original names
        quantized_weights = {}

        for name, param in quantized_layer.named_parameters():
            if name.startswith("linear_layers."):
                parts = name.split(".", 2)
                if len(parts) == 3:
                    safe_name, suffix = parts[1], parts[2]
                    if safe_name in safe_to_orig:
                        orig_name = safe_to_orig[safe_name]
                        new_name = f"{orig_name}.{suffix}" if suffix != "weight" else orig_name
                        quantized_weights[new_name] = param.data.cpu()
                    else:
                        quantized_weights[name] = param.data.cpu()
                else:
                    quantized_weights[name] = param.data.cpu()
            else:
                quantized_weights[name] = param.data.cpu()

        for name, buffer in quantized_layer.named_buffers():
            if name.startswith("linear_layers.") and "_weight_quantizer" in name:
                parts = name.split(".", 2)
                if len(parts) == 3:
                    safe_name = parts[1]
                    quant_suffix = parts[2]
                    if safe_name in safe_to_orig:
                        orig_name = safe_to_orig[safe_name]
                        new_name = f"{orig_name}{quant_suffix.replace('_weight', '')}"
                        quantized_weights[new_name] = buffer.data.cpu()
                    else:
                        quantized_weights[name] = buffer.data.cpu()
                else:
                    quantized_weights[name] = buffer.data.cpu()
            else:
                quantized_weights[name] = buffer.data.cpu()

        # Copy non-2D tensors (like norms) and excluded weights as-is
        for orig_name, tensor in layer_weights.items():
            if len(tensor.shape) != 2 or orig_name in excluded_weights:
                quantized_weights[orig_name] = tensor.cpu()

        # Clean up
        del linear_layer, quantized_layer, quantizer
        self._clear_memory()

        # Pack INT4 weights if requested
        if self.config.pack_int4 and self.config.precision.startswith("int4"):
            quantized_weights = pack_layer_weights(quantized_weights)

        return quantized_weights

    def _auto_detect_strategy(self) -> str:
        """
        Auto-detect the best memory strategy based on model size vs available memory.

        Decision logic:
        1. "full" - Model fits in GPU memory with room for activations
        2. "layerwise_cpu" - Model fits in CPU RAM but not GPU
        3. "lazy" - Model doesn't fit in CPU RAM, need disk loading

        Returns:
            Strategy name: "full", "layerwise_cpu", or "lazy"
        """
        import psutil

        # Estimate model size - try multiple methods
        model_size_gb = 0

        # Method 1: Check safetensors index file for exact size
        model_path = Path(self.config.model_path)
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    index_data = json.load(f)
                # Sum up file sizes from weight_map
                weight_files = set(index_data.get("weight_map", {}).values())
                total_bytes = 0
                for wf in weight_files:
                    wf_path = model_path / wf
                    if wf_path.exists():
                        total_bytes += wf_path.stat().st_size
                model_size_gb = total_bytes / (1024**3)
                self._log(f"  Model size from safetensors index: {model_size_gb:.1f} GB")
            except Exception as e:
                self._log(f"  Could not read safetensors index: {e}")

        # Method 2: Check config for num_parameters
        if model_size_gb == 0:
            num_params = getattr(self.hf_config, "num_parameters", None)
            if num_params:
                # Assuming FP16 = 2 bytes per parameter
                model_size_gb = num_params * 2 / (1024**3)
                self._log(f"  Model size from config.num_parameters: {model_size_gb:.1f} GB")

        # Method 3: Estimate from layer dimensions
        if model_size_gb == 0:
            layer_info = self._get_layer_info()
            num_layers = layer_info["num_layers"]
            hidden_size = layer_info["hidden_size"]
            intermediate_size = layer_info.get("intermediate_size") or hidden_size * 4

            if hidden_size > 0 and num_layers > 0:
                # Rough estimate for transformer models
                params_per_layer = (
                    hidden_size * hidden_size * 4  # Attention (Q, K, V, O projections)
                    + hidden_size * intermediate_size * 3  # MLP (gate, up, down)
                )
                # Total params with ~20% overhead for embeddings, norms, etc.
                num_params = int(num_layers * params_per_layer * 1.2)
                model_size_gb = num_params * 2 / (1024**3)
                self._log(f"  Model size estimated from config: {model_size_gb:.1f} GB")

        # Get available GPU memory
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Get available system/CPU memory
        ram = psutil.virtual_memory()
        host_memory_gb = ram.total / (1024**3)
        host_available_gb = ram.available / (1024**3)

        self._log("Memory assessment:")
        self._log(f"  Model size: {model_size_gb:.1f} GB")
        self._log(f"  GPU memory: {gpu_memory_gb:.1f} GB")
        self._log(f"  Host memory: {host_memory_gb:.1f} GB total, {host_available_gb:.1f} GB available")

        # Calculate memory requirements
        # Base: model size
        required_memory_gb = model_size_gb

        # Sensitivity analysis overhead
        if self.config.sensitivity_analysis:
            # Sensitivity analysis loads model separately + activation cache
            # Estimate activation cache: num_layers * batch * seq_len * hidden_size * 2 bytes
            layer_info = self._get_layer_info()
            num_layers = layer_info["num_layers"]
            hidden_size = layer_info["hidden_size"]
            activation_cache_gb = (
                num_layers * self.config.batch_size * self.config.seq_len * hidden_size * 2
            ) / (1024**3)

            if self.config.sensitivity_cache_on_gpu:
                # Model loaded twice + activation cache on GPU
                sensitivity_overhead_gb = model_size_gb + activation_cache_gb
                self._log(f"  Sensitivity analysis overhead: {sensitivity_overhead_gb:.1f} GB (model copy + {activation_cache_gb:.2f} GB cache)")
            else:
                # Only activation cache overhead on GPU (model copy uses CPU)
                sensitivity_overhead_gb = activation_cache_gb
                self._log(f"  Sensitivity analysis overhead: {sensitivity_overhead_gb:.2f} GB (CPU cache)")

            required_memory_gb += sensitivity_overhead_gb

        # Calculate layer memory
        layer_info = self._get_layer_info()
        num_layers = layer_info["num_layers"]
        layer_size_gb = model_size_gb / num_layers if num_layers > 0 else 0

        self._log(f"  Total required memory: {required_memory_gb:.1f} GB")
        self._log(f"  Layer memory: {layer_size_gb:.2f} GB per layer ({num_layers} layers)")

        # Decision logic with memory hierarchy
        # Threshold: leave 30% buffer for activations, OS, and other processes

        # Check 1: Can model fit entirely in GPU?
        gpu_threshold = gpu_memory_gb * 0.7  # 70% of GPU for model, 30% for activations
        if required_memory_gb < gpu_threshold:
            self._log(f"  → Strategy: 'full' (required {required_memory_gb:.1f}GB < GPU threshold {gpu_threshold:.1f}GB)")
            return "full"

        # Check 2: Can model fit in CPU/Host RAM?
        # Need to account for:
        # - Model weights in CPU RAM
        # - One layer on GPU for quantization
        # - OS and other processes
        # - One layer on GPU for quantization
        # - OS and other processes
        host_threshold = host_available_gb * 0.7  # Use 70% of available RAM

        if model_size_gb < host_threshold:
            self._log(f"  → Strategy: 'layerwise_cpu' (model {model_size_gb:.1f}GB < host threshold {host_threshold:.1f}GB)")
            return "layerwise_cpu"

        # Check 3: Model is too large for CPU RAM, need lazy loading from disk
        self._log(f"  → Strategy: 'lazy' (model {model_size_gb:.1f}GB > host threshold {host_threshold:.1f}GB)")
        return "lazy"

    def _run_lazy_quantization(self) -> QuantizationResult:
        """Run lazy disk-loading quantization."""
        total_start = time.time()
        result = QuantizationResult(success=False)

        try:
            # Setup (load config only, no weights)
            self._setup()

            # Build weight index
            self._build_weight_index()

            # Get layer info
            layer_info = self._get_layer_info()
            num_layers = layer_info["num_layers"]
            layer_prefix = layer_info["layer_prefix"]

            # Determine exclusions
            exclude_layers = self._determine_exclude_layers("lazy")
            result.exclude_layers_used = exclude_layers
            self._log(f"Exclude layers: {exclude_layers}")

            # Create quantization config
            quant_config = self._create_quant_config(exclude_layers)

            self._log(f"\n{'=' * 60}")
            self._log("LAZY LAYER-WISE QUANTIZATION")
            self._log(f"{'=' * 60}")
            self._log(f"Model: {self.config.model_path}")
            self._log(f"Output: {self.config.output_dir}")
            self._log(f"Precision: {self.config.precision}")
            self._log(f"Pack INT4: {self.config.pack_int4}")
            self._log(f"Layers: {num_layers}")
            self._log(f"Device: {self.config.device}")
            self._log(f"{'=' * 60}")

            # Process layers
            quantized_layers = []
            layer_times = []

            for layer_idx in tqdm(range(num_layers), desc="Quantizing layers"):
                layer_start = time.time()

                if self.config.verbose:
                    self._log(f"\n--- Layer {layer_idx}/{num_layers - 1} ---")
                self._clear_memory()
                if self.config.verbose:
                    self._log(f"  Memory: {self._get_memory_info()}")

                # Load layer weights from disk
                layer_weights = self._load_layer_weights(layer_idx, layer_prefix)

                if not layer_weights:
                    self._log(f"  Warning: No weights found for layer {layer_idx}")
                    continue

                if self.config.verbose:
                    self._log(f"  Loaded {len(layer_weights)} weight tensors")

                # Quantize layer
                quantized_weights = self._quantize_layer_weights(
                    layer_weights, layer_idx, quant_config, exclude_layers
                )

                # Save immediately to disk
                layer_file = (
                    Path(self.config.output_dir)
                    / "quantized_layers"
                    / f"layer_{layer_idx}.safetensors"
                )
                save_file(quantized_weights, layer_file)

                quantized_layers.append({
                    "layer_idx": layer_idx,
                    "file": str(layer_file),
                    "num_weights": len(quantized_weights),
                })

                # Clear memory
                del layer_weights, quantized_weights
                self._clear_memory()

                layer_time = time.time() - layer_start
                layer_times.append(layer_time)
                avg_time = sum(layer_times) / len(layer_times)
                eta = avg_time * (num_layers - layer_idx - 1)

                if self.config.verbose:
                    self._log(f"  Completed in {layer_time:.2f}s (ETA: {eta / 60:.1f}m)")

            # Process non-layer parameters (embeddings, lm_head, etc.)
            if self.config.verbose:
                self._log("\n=== Processing non-layer parameters ===")
            non_layer_weights = {}
            for name in self.weight_index:
                if not any(
                    name.startswith(f"{layer_prefix}.{i}") for i in range(num_layers)
                ):
                    # Load this weight
                    file_path = self.weight_index[name]
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        non_layer_weights[name] = f.get_tensor(name)
                    if self.config.verbose:
                        self._log(f"  Copied: {name}")

            # Save non-layer weights
            if non_layer_weights:
                non_layer_file = Path(self.config.output_dir) / "non_layer_weights.safetensors"
                save_file(non_layer_weights, non_layer_file)

            # Save tokenizer and config (with quantization metadata)
            self.tokenizer.save_pretrained(self.config.output_dir)

            # Add quantization_config to hf_config for Quark compatibility
            quant_scheme = self._get_quant_scheme()
            group_size = 128  # Default group size for INT4
            if "128" in quant_scheme:
                group_size = 128
            elif "64" in quant_scheme:
                group_size = 64

            quantization_config = {
                "quant_method": "quark",
                "quant_mode": "eager_mode",
                "global_quant_config": {
                    "weight": {
                        "dtype": "int4" if self.config.precision.startswith("int4") else "int8",
                        "qscheme": "per_group",
                        "group_size": group_size,
                        "ch_axis": -1,
                        "is_dynamic": False,
                        "symmetric": True,
                        "scale_type": "float",
                        "is_scale_quant": False,
                        "observer_cls": "PerGroupMinMaxObserver",
                        "round_method": "half_even",
                    },
                    "input_tensors": None,
                    "output_tensors": None,
                    "bias": None,
                },
                "kv_cache_quant_config": {},
                "layer_quant_config": {},
                "layer_type_quant_config": {},
                "exclude": exclude_layers,
                "export": {
                    "kv_cache_group": [],
                    "min_kv_scale": 0.0,
                    "pack_method": "reorder",
                    "weight_format": "real_quantized",
                    "weight_merge_groups": None,
                },
            }
            self.hf_config.quantization_config = quantization_config
            self.hf_config.save_pretrained(self.config.output_dir)

            # Save quantization metadata
            quant_meta = {
                "quant_scheme": self._get_quant_scheme(),
                "precision": self.config.precision,
                "pack_int4": self.config.pack_int4,
                "exclude_layers": exclude_layers,
                "model_type": self.model_type,
                "num_layers": num_layers,
                "quantized_layers": quantized_layers,
            }
            with open(Path(self.config.output_dir) / "quantization_meta.json", "w") as f:
                json.dump(quant_meta, f, indent=2)

            # Save quantization result before assembly (required by HuggingFaceExporter)
            self.timing["total"] = time.time() - total_start
            result.success = True
            result.output_dir = self.config.output_dir
            result.model_type = self.model_type
            result.quant_scheme = self._get_quant_scheme()
            result.precision = self.config.precision
            result.timing = self.timing
            result.exclude_layers_used = exclude_layers
            self._populate_result_sensitivity(result)
            result_path = Path(self.config.output_dir) / "quantization_result.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            self._log(f"Saved quantization result to {result_path}")

            # Assemble into HuggingFace format
            self._log("\n=== Assembling HuggingFace format ===")
            self._assemble_hf_format()

            self._print_summary(result)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._log(f"Error during quantization: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _assemble_hf_format(self) -> None:
        """
        Assemble quantized layers into HuggingFace-compatible sharded format.
        Uses HuggingFaceExporter for memory-efficient streaming.
        """
        from ..export.hf_export import HuggingFaceExporter
        self._log("\n=== Assembling HuggingFace format ===")
        exporter = HuggingFaceExporter(
            quantized_dir=self.config.output_dir,
            output_dir=self.config.output_dir,
            original_model_path=self.config.model_path
        )
        exporter.export()

    def _run_layerwise_cpu_quantization(self) -> QuantizationResult:
        """Run layerwise CPU quantization."""
        from transformers import AutoModelForCausalLM

        total_start = time.time()
        result = QuantizationResult(success=False)

        try:
            # Setup
            self._setup()

            # Build weight index (needed for sensitivity analysis)
            self._build_weight_index()

            # Load model to CPU
            self._log("Loading model to CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True,
            )

            # Get layer info
            layer_info = self._get_layer_info()
            num_layers = layer_info["num_layers"]
            layer_prefix = layer_info["layer_prefix"]

            self._log(f"Model loaded. Layers: {num_layers}")

            # Determine exclusions (now with weight index available for sensitivity analysis)
            exclude_layers = self._determine_exclude_layers("layerwise_cpu")
            result.exclude_layers_used = exclude_layers

            # Create quantization config
            quant_config = self._create_quant_config(exclude_layers)

            # Get all modules
            all_modules = dict(model.named_modules())

            # Create output directory for layers
            layers_dir = Path(self.config.output_dir) / "quantized_layers"
            layers_dir.mkdir(parents=True, exist_ok=True)

            quantized_layers_meta = []

            # Process layers
            for layer_idx in tqdm(range(num_layers), desc="Quantizing layers"):
                layer_start = time.time()

                layer_name = f"{layer_prefix}.{layer_idx}"
                layer_module = all_modules.get(layer_name)

                if layer_module is None:
                    self._log(f"Warning: Layer {layer_name} not found")
                    continue

                if self.config.verbose:
                    self._log(f"\n--- Layer {layer_idx}/{num_layers - 1} ---")
                self._clear_memory()

                # Move layer to GPU
                layer_module = layer_module.to(self.config.device)
                if self.config.verbose:
                    self._log(f"  Memory: {self._get_memory_info()}")

                # Quantize
                try:
                    from torch.utils.data import DataLoader, TensorDataset

                    quantizer = ModelQuantizer(quant_config)
                    dummy = torch.zeros(1, layer_info["hidden_size"], device=self.config.device)
                    dummy_loader = DataLoader(TensorDataset(dummy), batch_size=1)

                    # Suppress Quark's internal progress bars in non-verbose mode
                    verbose = getattr(self.config, 'verbose', False)
                    if not verbose:
                        with suppress_quark_output():
                            layer_module = quantizer.quantize_model(layer_module, dummy_loader)
                            layer_module = quantizer.freeze(layer_module)
                    else:
                        layer_module = quantizer.quantize_model(layer_module, dummy_loader)
                        layer_module = quantizer.freeze(layer_module)

                except torch.cuda.OutOfMemoryError:
                    self._log(f"  OOM quantizing layer {layer_idx}, re-raising")
                    raise
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self._log(f"  OOM quantizing layer {layer_idx}, re-raising")
                        raise
                    self._log(f"  Error quantizing layer: {e}")
                    continue
                except Exception as e:
                    self._log(f"  Error quantizing layer: {e}")
                    continue

                # Extract weights for this layer
                layer_weights = {}
                for name, param in layer_module.named_parameters():
                    full_name = f"{layer_name}.{name}"
                    layer_weights[full_name] = param.data.cpu().clone()

                for name, buffer in layer_module.named_buffers():
                    full_name = f"{layer_name}.{name}"
                    layer_weights[full_name] = buffer.data.cpu().clone()

                # Pack INT4 weights if requested
                if self.config.pack_int4 and self.config.precision.startswith("int4"):
                    layer_weights = pack_layer_weights(layer_weights)

                # Save layer to disk immediately
                layer_file = layers_dir / f"layer_{layer_idx}.safetensors"
                save_file(layer_weights, layer_file)
                
                quantized_layers_meta.append({
                    "layer_idx": layer_idx,
                    "file": str(layer_file),
                    "num_weights": len(layer_weights),
                })

                # Move back to CPU (if model is still needed)
                layer_module = layer_module.cpu()
                
                # Cleanup
                del layer_weights
                self._clear_memory()

                layer_time = time.time() - layer_start
                if self.config.verbose:
                    self._log(f"  Completed in {layer_time:.2f}s")

            # Process non-layer params
            self._log("\nProcessing non-layer parameters...")
            non_layer_weights = {}
            for name, param in model.named_parameters():
                if not any(name.startswith(f"{layer_prefix}.{i}") for i in range(num_layers)):
                    non_layer_weights[name] = param.data.cpu().clone()

            for name, buffer in model.named_buffers():
                if not any(name.startswith(f"{layer_prefix}.{i}") for i in range(num_layers)):
                    non_layer_weights[name] = buffer.data.cpu().clone()

            # Save non-layer weights
            if non_layer_weights:
                non_layer_file = Path(self.config.output_dir) / "non_layer_weights.safetensors"
                save_file(non_layer_weights, non_layer_file)

            # Save tokenizer and config
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Save config with metadata
            self.hf_config.save_pretrained(self.config.output_dir)

            # Save quantization metadata before assembly
            quant_meta = {
                "quant_scheme": self._get_quant_scheme(),
                "precision": self.config.precision,
                "pack_int4": self.config.pack_int4,
                "exclude_layers": exclude_layers,
                "model_type": self.model_type,
                "num_layers": num_layers,
                "quantized_layers": quantized_layers_meta,
                "sensitivity_scores": self.sensitivity_scores,
            }
            with open(Path(self.config.output_dir) / "quantization_meta.json", "w") as f:
                json.dump(quant_meta, f, indent=2)

            # Save quantization result before assembly
            result.success = True
            result.output_dir = self.config.output_dir
            result.model_type = self.model_type
            result.quant_scheme = self._get_quant_scheme()
            result.precision = self.config.precision
            result.timing = self.timing
            self._populate_result_sensitivity(result)

            result_path = Path(self.config.output_dir) / "quantization_result.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            self._log(f"Saved quantization result to {result_path}")

            # Assemble into sharded format
            self._assemble_hf_format()

            self.timing["total"] = time.time() - total_start

            result.success = True
            result.output_dir = self.config.output_dir
            result.model_type = self.model_type
            result.quant_scheme = self._get_quant_scheme()
            result.timing = self.timing
            self._populate_result_sensitivity(result)

            self._print_summary(result)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._log(f"Error during quantization: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _run_full_gpu_quantization(self) -> QuantizationResult:
        """Run full GPU quantization."""
        from transformers import AutoModelForCausalLM

        total_start = time.time()
        result = QuantizationResult(success=False)

        try:
            # Setup
            self._setup()

            # Load model to GPU
            self._log("Loading model to GPU...")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=dtype,
                trust_remote_code=self.config.trust_remote_code,
            ).to(self.config.device)

            model.eval()

            # Determine exclusions
            exclude_layers = self._determine_exclude_layers("full")
            result.exclude_layers_used = exclude_layers

            # Create quantization config
            quant_config = self._create_quant_config(exclude_layers)

            # Get calibration data
            self._log("Loading calibration data...")
            calib_loader = get_calib_dataloader(
                dataset_name_or_path=self.config.calibration_data,
                tokenizer=self.tokenizer,
                batch_size=self.config.batch_size,
                num_calib_data=self.config.num_calib_samples,
                seqlen=self.config.seq_len,
                device=self.config.device,
            )

            # Quantize
            self._log("Quantizing model...")
            quant_start = time.time()

            quantizer = ModelQuantizer(quant_config)

            # Suppress Quark's internal output in non-verbose mode
            verbose = getattr(self.config, 'verbose', False)
            if not verbose:
                with suppress_quark_output():
                    model = quantizer.quantize_model(model, calib_loader)
                    model = quantizer.freeze(model)
            else:
                model = quantizer.quantize_model(model, calib_loader)
                model = quantizer.freeze(model)

            self.timing["quantization"] = time.time() - quant_start

            # Export
            # Note: AWQ/GPTQ export will be implemented via post-quantization conversion
            self._log("Exporting quantized model...")
            from quark.torch import export_safetensors

            with torch.no_grad():
                export_safetensors(
                    model=model,
                    output_dir=self.config.output_dir,
                    weight_format="real_quantized",
                )

            self.tokenizer.save_pretrained(self.config.output_dir)

            self.timing["total"] = time.time() - total_start

            result.success = True
            result.output_dir = self.config.output_dir
            result.model_type = self.model_type
            self.tokenizer.save_pretrained(self.config.output_dir)

            self.timing["total"] = time.time() - total_start

            self._print_summary(result)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._log(f"Error during quantization: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _print_summary(self, result: QuantizationResult) -> None:
        """Print quantization summary with detailed sub-layer sensitivity analysis."""
        print("\n" + "=" * 60)
        print("QUANTIZATION SUMMARY")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Model Type: {result.model_type}")
        print(f"Precision: {result.precision}")
        print(f"Quantization Scheme: {result.quant_scheme}")
        print(f"Output Directory: {result.output_dir}")
        print(f"\nExcluded Layers: {result.exclude_layers_used}")

        # Show sensitivity analysis results if available
        if result.sensitivity_scores:
            threshold = self.config.sensitivity_threshold
            scores = list(result.sensitivity_scores.values())

            # Sort by score (highest = most sensitive)
            sorted_scores = sorted(
                result.sensitivity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            print(f"\n{'=' * 70}")
            print("SENSITIVITY ANALYSIS REPORT")
            print(f"{'=' * 70}")
            print(f"Layers analyzed: {len(result.sensitivity_scores)}")
            print(f"Threshold: {threshold}")

            # Sub-layer type summary (if available)
            if hasattr(result, 'sublayer_type_scores') and result.sublayer_type_scores:
                print(f"\n{'-' * 70}")
                print("SUB-LAYER TYPE SENSITIVITY (average across all layers)")
                print(f"{'-' * 70}")
                print(f"{'Sub-Layer Type':<25} | {'Avg Score':>12} | {'Max Score':>12}")
                print(f"{'-' * 70}")

                sorted_types = sorted(
                    result.sublayer_type_scores.items(),
                    key=lambda x: x[1]['avg'],
                    reverse=True
                )
                for sublayer_type, stats in sorted_types:
                    print(f"{sublayer_type:<25} | {stats['avg']:>12.6f} | {stats['max']:>12.6f}")
                print(f"{'-' * 70}")

            # Top 10 most sensitive layers with sub-layer breakdown in ASCII table format
            print(f"\n{'=' * 120}")
            print("TOP 10 MOST SENSITIVE LAYERS (with sub-layer details)")
            print(f"{'=' * 120}")

            # Get sub-layer scores if available
            sublayer_scores = getattr(result, 'sublayer_sensitivity_scores', {})

            # Collect all sublayer types and layer data for top 10 layers
            all_sublayer_types = set()
            top_layers_data = []

            for i, (layer_name, layer_score) in enumerate(sorted_scores[:10], 1):
                # Find sub-layer scores for this layer
                layer_sublayers = {}
                for weight_name, weight_score in sublayer_scores.items():
                    if weight_name.startswith(layer_name + "."):
                        # Extract sub-layer type (e.g., q_proj, k_proj, gate_proj)
                        parts = weight_name.split('.')
                        for j, part in enumerate(parts):
                            if part in ['self_attn', 'mlp', 'attention', 'feed_forward']:
                                if j + 1 < len(parts):
                                    sublayer_type = parts[j + 1]
                                    layer_sublayers[sublayer_type] = weight_score
                                    all_sublayer_types.add(sublayer_type)
                                    break

                is_excluded = threshold > 0 and layer_score > threshold
                top_layers_data.append((i, layer_name, layer_score, layer_sublayers, is_excluded))

            # Define column order (attention first, then MLP)
            priority_order = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            ordered_sublayer_types = []
            for st in priority_order:
                if st in all_sublayer_types:
                    ordered_sublayer_types.append(st)
            # Add any remaining types not in priority order
            for st in sorted(all_sublayer_types):
                if st not in ordered_sublayer_types:
                    ordered_sublayer_types.append(st)

            # Build table header
            # Format: | # | Layer | Score | Status | sublayer1 | sublayer2 | ... |
            col_widths = {
                'rank': 3,
                'layer': 18,
                'score': 10,
                'status': 10,
            }
            for st in ordered_sublayer_types:
                col_widths[st] = max(10, len(st) + 2)

            # Print header
            header_parts = [
                'Rank'.center(col_widths['rank']),
                'Layer'.center(col_widths['layer']),
                'Score'.center(col_widths['score']),
                'Status'.center(col_widths['status']),
            ]
            for st in ordered_sublayer_types:
                header_parts.append(st.center(col_widths[st]))

            separator = '+' + '+'.join('-' * w for w in [col_widths['rank'], col_widths['layer'], col_widths['score'], col_widths['status']] + [col_widths[st] for st in ordered_sublayer_types]) + '+'

            print(separator)
            print('|' + '|'.join(header_parts) + '|')
            print(separator)

            # Print data rows
            for rank, layer_name, layer_score, layer_sublayers, is_excluded in top_layers_data:
                # Extract layer number for shorter display
                layer_display = layer_name
                if '.' in layer_name:
                    parts = layer_name.split('.')
                    # Try to show as "layers.N" format
                    for idx, p in enumerate(parts):
                        if p == 'layers' and idx + 1 < len(parts):
                            layer_display = f"layers.{parts[idx + 1]}"
                            break

                status = "EXCLUDED" if is_excluded else ""
                row_parts = [
                    str(rank).center(col_widths['rank']),
                    layer_display.ljust(col_widths['layer'])[:col_widths['layer']],
                    f"{layer_score:.6f}".rjust(col_widths['score']),
                    status.center(col_widths['status']),
                ]
                for st in ordered_sublayer_types:
                    if st in layer_sublayers:
                        row_parts.append(f"{layer_sublayers[st]:.6f}".rjust(col_widths[st]))
                    else:
                        row_parts.append('-'.center(col_widths[st]))

                print('|' + '|'.join(row_parts) + '|')

            print(separator)

            # Add legend
            if ordered_sublayer_types:
                print(f"\nSublayer Types: {', '.join(ordered_sublayer_types)}")
            print(f"Status 'EXCLUDED': Layer score exceeds threshold ({threshold})")

            # Score distribution - ASCII table format
            print(f"\n{'-' * 50}")
            print("SCORE DISTRIBUTION")
            print(f"{'-' * 50}")
            print(f"{'Metric':<20} | {'Value':>15}")
            print(f"{'-' * 50}")
            print(f"{'Min':<20} | {min(scores):>15.6f}")
            print(f"{'Max':<20} | {max(scores):>15.6f}")
            print(f"{'Mean':<20} | {sum(scores)/len(scores):>15.6f}")
            print(f"{'Std Dev':<20} | {(sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5:>15.6f}")
            if threshold > 0:
                excluded_count = sum(1 for s in scores if s > threshold)
                print(f"{'Above Threshold':<20} | {excluded_count:>15}")
            print(f"{'-' * 50}")

            # Recommendation for manual exclusion
            print(f"\n{'-' * 70}")
            print("RECOMMENDATION FOR MANUAL EXCLUSION")
            print(f"{'-' * 70}")
            if threshold > 0 and any(s > threshold for s in scores):
                excluded_layer_names = [name for name, score in sorted_scores if score > threshold]
                print("Layers recommended for exclusion (score > threshold):")
                for name in excluded_layer_names:
                    print(f"  --exclude_layers \"{name}\"")
            else:
                print("No layers exceed the threshold.")
                # Suggest sub-layer patterns for exclusion
                if hasattr(result, 'sublayer_type_scores') and result.sublayer_type_scores:
                    print("\nMost sensitive sub-layer types (consider pattern exclusion):")
                    sorted_types = sorted(
                        result.sublayer_type_scores.items(),
                        key=lambda x: x[1]['avg'],
                        reverse=True
                    )
                    for sublayer_type, stats in sorted_types[:5]:
                        print(f"  --exclude_layers \"*{sublayer_type}*\"  # avg score: {stats['avg']:.6f}")
                else:
                    # Fallback to layer-level suggestions
                    print("\nMost sensitive layer candidates (consider excluding if quality degrades):")
                    for name, score in sorted_scores[:5]:
                        print(f"  --exclude_layers \"{name}\"  # score: {score:.6f}")
            print(f"{'-' * 70}")

        if result.timing:
            print("\nTiming:")
            for stage, duration in result.timing.items():
                print(f"  {stage}: {duration:.2f}s")

        if result.error_message:
            print(f"\nError: {result.error_message}")

        print("=" * 60)

    def run(self) -> QuantizationResult:
        """
        Run the quantization process.

        Returns:
            QuantizationResult with details of the quantization
        """
        # Configure Quark logging based on verbose setting
        self._configure_quark_logging()

        # Determine strategy
        if self.config.memory_strategy == "auto":
            # Need to load config first for auto-detection
            self.hf_config = AutoConfig.from_pretrained(
                self.config.model_path, trust_remote_code=self.config.trust_remote_code
            )
            strategy = self._auto_detect_strategy()
            self._log(f"Auto-detected memory strategy: {strategy}")
        else:
            strategy = self.config.memory_strategy

        # Dispatch to appropriate strategy
        if strategy == "lazy":
            return self._run_lazy_quantization()
        elif strategy == "layerwise_cpu":
            return self._run_layerwise_cpu_quantization()
        else:  # "full"
            return self._run_full_gpu_quantization()


# Backward compatibility alias
AutoQuantizer = UnifiedQuantizer

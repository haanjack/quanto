"""
Unified Quantizer: Single class supporting all quantization strategies.

This module consolidates the functionality of:
- AutoQuantizer (full GPU quantization)
- IterativeQuantizer (iterative exclusion discovery)
- LayerwiseQuantizer (CPU model, GPU quantization)
- LazyLayerwiseQuantizer (lazy disk loading with INT4 packing)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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

from ..constants import PRECISION_TO_SCHEME, SUPPORTED_PRECISIONS
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

    def _log(self, message: str) -> None:
        """Print log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

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
        """Get layer information from config."""
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
                with safe_open(sf_file, framework="pt") as f:
                    for key in f.keys():
                        self.weight_index[key] = str(sf_file)

        self._log(f"Found {len(self.safetensors_files)} safetensors files")
        self._log(f"Indexed {len(self.weight_index)} weight tensors")

    def _determine_exclude_layers(self) -> list[str]:
        """Determine layers to exclude based on config and model analysis."""
        if self.config.exclude_layers is not None:
            return self.config.exclude_layers

        # Default exclusions
        exclude = ["lm_head"]

        # Add standard patterns
        exclude.extend(["*embed*", "*norm*"])

        if self.config.aggressive_exclusion:
            exclude.extend(["*gate*"])

        # Add sensitivity-based exclusions if threshold is set
        if self.config.sensitivity_threshold > 0:
            self._log("Running sensitivity analysis for exclusion detection...")
            sensitive_layers = self._analyze_sensitive_layers()
            exclude.extend(sensitive_layers)

        # Remove duplicates
        return list(set(exclude))

    def _analyze_sensitive_layers(self) -> list[str]:
        """Analyze layers for sensitivity to quantization."""
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

    def _get_layer_weight_names(self, layer_idx: int, layer_prefix: str) -> list[str]:
        """Get all weight names for a specific layer."""
        layer_pattern = f"{layer_prefix}.{layer_idx}."
        return [k for k in self.weight_index.keys() if k.startswith(layer_pattern)]

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

        if self.template:
            return self.template.get_config(
                scheme=quant_scheme,
                exclude_layers=exclude_layers,
            )
        else:
            return QConfig(
                global_quant_config=QLayerConfig(
                    weight=Int4PerGroupSpec(group_size=128).to_quantization_spec()
                ),
                exclude=exclude_layers,
            )

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

                for name, tensor in weights.items():
                    # Only process 2D tensors (Linear weights)
                    if len(tensor.shape) == 2:
                        out_features, in_features = tensor.shape
                        linear = nn.Linear(in_features, out_features, bias=False)
                        linear.weight.data = tensor
                        # Use sanitized name for ModuleDict
                        safe_name = name.replace(".", "_")
                        self.linear_layers[safe_name] = linear

            def forward(self, x):
                return x  # Dummy forward

        # Check if any weights should be excluded
        import fnmatch

        excluded_weights = set()
        for name in layer_weights.keys():
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
        linear_layer = linear_layer.to(self.config.device)

        # Create quantizer
        quantizer = ModelQuantizer(quant_config)

        # Create dummy dataloader for weight-only quantization
        from torch.utils.data import DataLoader, TensorDataset

        dummy = torch.zeros(1, 1, device=self.config.device)
        dummy_loader = DataLoader(TensorDataset(dummy), batch_size=1)

        # Quantize
        quantized_layer = quantizer.quantize_model(linear_layer, dummy_loader)
        quantized_layer = quantizer.freeze(quantized_layer)

        # Extract quantized weights with original names
        quantized_weights = {}
        safe_to_orig = {
            name.replace(".", "_"): name
            for name in weights_to_quantize
            if len(weights_to_quantize[name].shape) == 2
        }

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

        self._log(f"Memory assessment:")
        self._log(f"  Model size: {model_size_gb:.1f} GB")
        self._log(f"  GPU memory: {gpu_memory_gb:.1f} GB")
        self._log(f"  Host memory: {host_memory_gb:.1f} GB total, {host_available_gb:.1f} GB available")

        # Decision logic with memory hierarchy
        # Threshold: leave 30% buffer for activations, OS, and other processes

        # Check 1: Can model fit entirely in GPU?
        gpu_threshold = gpu_memory_gb * 0.7  # 70% of GPU for model, 30% for activations
        if model_size_gb < gpu_threshold:
            self._log(f"  → Strategy: 'full' (model {model_size_gb:.1f}GB < GPU threshold {gpu_threshold:.1f}GB)")
            return "full"

        # Check 2: Can model fit in CPU/Host RAM?
        # Need to account for:
        # - Model weights in CPU RAM
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
            exclude_layers = self._determine_exclude_layers()
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

                self._log(f"\n--- Layer {layer_idx}/{num_layers - 1} ---")
                self._clear_memory()
                self._log(f"  Memory: {self._get_memory_info()}")

                # Load layer weights from disk
                layer_weights = self._load_layer_weights(layer_idx, layer_prefix)

                if not layer_weights:
                    self._log(f"  Warning: No weights found for layer {layer_idx}")
                    continue

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

                self._log(f"  Completed in {layer_time:.2f}s (ETA: {eta / 60:.1f}m)")

            # Process non-layer parameters (embeddings, lm_head, etc.)
            self._log("\n=== Processing non-layer parameters ===")
            non_layer_weights = {}
            for name in self.weight_index.keys():
                if not any(
                    name.startswith(f"{layer_prefix}.{i}") for i in range(num_layers)
                ):
                    # Load this weight
                    file_path = self.weight_index[name]
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        non_layer_weights[name] = f.get_tensor(name)
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

            # Assemble into HuggingFace format
            self._log("\n=== Assembling HuggingFace format ===")
            self._assemble_hf_format()

            self.timing["total"] = time.time() - total_start

            result.success = True
            result.output_dir = self.config.output_dir
            result.model_type = self.model_type
            result.quant_scheme = self._get_quant_scheme()
            result.precision = self.config.precision
            result.timing = self.timing

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

        Combines:
        - quantized_layers/layer_*.safetensors
        - non_layer_weights.safetensors

        Into:
        - model-00001-of-XXXXX.safetensors (sharded, max 5GB each)
        - model.safetensors.index.json
        """
        output_dir = Path(self.config.output_dir)
        layers_dir = output_dir / "quantized_layers"
        non_layer_file = output_dir / "non_layer_weights.safetensors"

        # Collect all weights
        all_weights = {}
        weight_map = {}

        # Load layer files
        layer_files = sorted(
            layers_dir.glob("layer_*.safetensors"),
            key=lambda x: int(x.stem.split("_")[1])
        )
        self._log(f"Loading {len(layer_files)} layer files...")

        for layer_file in tqdm(layer_files, desc="Loading layers"):
            with safe_open(layer_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)

        # Load non-layer weights
        if non_layer_file.exists():
            self._log("Loading non-layer weights...")
            with safe_open(non_layer_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)

        self._log(f"Total tensors: {len(all_weights)}")

        # Calculate shards (max 5GB per shard)
        MAX_SHARD_SIZE = 5 * 1024**3  # 5GB

        # Group weights by size into shards
        shards = []
        current_shard = {}
        current_size = 0

        for name, tensor in all_weights.items():
            tensor_size = tensor.numel() * tensor.element_size()

            if current_size + tensor_size > MAX_SHARD_SIZE and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0

            current_shard[name] = tensor
            current_size += tensor_size

        if current_shard:
            shards.append(current_shard)

        num_shards = len(shards)
        self._log(f"Creating {num_shards} shard files...")

        # Save shards and build index
        for i, shard in enumerate(tqdm(shards, desc="Saving shards"), 1):
            shard_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
            shard_file = output_dir / shard_name
            save_file(shard, shard_file)

            for name in shard.keys():
                weight_map[name] = shard_name

        # Create index file
        index = {
            "metadata": {
                "total_size": sum(
                    t.numel() * t.element_size() for t in all_weights.values()
                )
            },
            "weight_map": weight_map
        }

        index_file = output_dir / "model.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

        self._log(f"Created {num_shards} shards with index")

        # Clean up temporary files (optional - keep for debugging)
        # shutil.rmtree(layers_dir)
        # non_layer_file.unlink()

    def _run_layerwise_cpu_quantization(self) -> QuantizationResult:
        """Run layerwise CPU quantization."""
        from transformers import AutoModelForCausalLM

        total_start = time.time()
        result = QuantizationResult(success=False)

        try:
            # Setup
            self._setup()

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

            # Determine exclusions
            exclude_layers = self._determine_exclude_layers()
            result.exclude_layers_used = exclude_layers

            # Create quantization config
            quant_config = self._create_quant_config(exclude_layers)

            # Get all modules
            all_modules = dict(model.named_modules())

            # Track quantized weights
            quantized_state_dict = {}

            # Process layers
            for layer_idx in tqdm(range(num_layers), desc="Quantizing layers"):
                layer_start = time.time()

                layer_name = f"{layer_prefix}.{layer_idx}"
                layer_module = all_modules.get(layer_name)

                if layer_module is None:
                    self._log(f"Warning: Layer {layer_name} not found")
                    continue

                self._log(f"\n--- Layer {layer_idx}/{num_layers - 1} ---")
                self._clear_memory()

                # Move layer to GPU
                layer_module = layer_module.to(self.config.device)
                self._log(f"  Memory: {self._get_memory_info()}")

                # Quantize
                try:
                    from torch.utils.data import DataLoader, TensorDataset

                    quantizer = ModelQuantizer(quant_config)
                    dummy = torch.zeros(1, layer_info["hidden_size"], device=self.config.device)
                    dummy_loader = DataLoader(TensorDataset(dummy), batch_size=1)

                    layer_module = quantizer.quantize_model(layer_module, dummy_loader)
                    layer_module = quantizer.freeze(layer_module)

                except Exception as e:
                    self._log(f"  Error quantizing layer: {e}")

                # Extract weights
                for name, param in layer_module.named_parameters():
                    full_name = f"{layer_name}.{name}"
                    quantized_state_dict[full_name] = param.data.cpu().clone()

                for name, buffer in layer_module.named_buffers():
                    full_name = f"{layer_name}.{name}"
                    quantized_state_dict[full_name] = buffer.data.cpu().clone()

                # Move back to CPU
                layer_module = layer_module.cpu()
                del layer_module
                self._clear_memory()

                layer_time = time.time() - layer_start
                self._log(f"  Completed in {layer_time:.2f}s")

            # Process non-layer params
            self._log("\nProcessing non-layer parameters...")
            for name, param in model.named_parameters():
                if not any(name.startswith(f"{layer_prefix}.{i}") for i in range(num_layers)):
                    quantized_state_dict[name] = param.data.clone()

            for name, buffer in model.named_buffers():
                if not any(name.startswith(f"{layer_prefix}.{i}") for i in range(num_layers)):
                    quantized_state_dict[name] = buffer.data.clone()

            # Save
            self._log("\nSaving quantized model...")
            output_file = os.path.join(self.config.output_dir, "model.safetensors")
            save_file(quantized_state_dict, output_file)

            self.tokenizer.save_pretrained(self.config.output_dir)
            self.hf_config.save_pretrained(self.config.output_dir)

            self.timing["total"] = time.time() - total_start

            result.success = True
            result.output_dir = self.config.output_dir
            result.model_type = self.model_type
            result.quant_scheme = self._get_quant_scheme()
            result.timing = self.timing

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
            exclude_layers = self._determine_exclude_layers()
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
            model = quantizer.quantize_model(model, calib_loader)
            model = quantizer.freeze(model)

            self.timing["quantization"] = time.time() - quant_start

            # Export
            self._log("Exporting quantized model...")
            from quark.torch import export_safetensors

            with torch.no_grad():
                export_safetensors(
                    model=model,
                    output_dir=self.config.output_dir,
                    custom_mode="quark",
                    weight_format="real_quantized",
                )

            self.tokenizer.save_pretrained(self.config.output_dir)

            self.timing["total"] = time.time() - total_start

            result.success = True
            result.output_dir = self.config.output_dir
            result.model_type = self.model_type
            result.quant_scheme = self._get_quant_scheme()
            result.precision = self.config.precision
            result.timing = self.timing

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
        print(f"Success: {result.success}")
        print(f"Model Type: {result.model_type}")
        print(f"Precision: {result.precision}")
        print(f"Quantization Scheme: {result.quant_scheme}")
        print(f"Output Directory: {result.output_dir}")
        print(f"\nExcluded Layers: {result.exclude_layers_used}")

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

"""
Lazy Layer-wise Quantization for Large LLMs

This module implements memory-efficient layer-by-layer quantization that:
1. Loads model config and architecture only (no weights)
2. Loads layer weights on-demand from safetensors files
3. Quantizes each layer on GPU
4. Saves quantized layer immediately
5. Releases memory before next layer

This approach allows quantization of models larger than available CPU RAM.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))
sys.path.insert(0, str(Path(__file__).parent))

from int4_pack import pack_int4_to_int32, quantize_to_int4
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from transformers import AutoConfig, AutoTokenizer


class LazyLayerwiseQuantizer:
    """
    Memory-efficient layer-wise quantizer using lazy weight loading.
    """

    PRECISION_TO_SCHEME = {
        "int8": "int8",
        "int4": "int4_wo_128",
        "int4_64": "int4_wo_64",
        "int4_32": "int4_wo_32",
        "fp8": "fp8",
    }

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        precision: str = "int4",
        calibration_data: str = "pileval",
        num_calib_samples: int = 16,
        seq_len: int = 512,
        device: str = "cuda",
        exclude_layers: list[str] | None = None,
        trust_remote_code: bool = True,
        sensitivity_threshold: float = 0.0,
        batch_size: int = 4,  # Number of layers to process in parallel
        pack_int4: bool = True,  # Pack INT4 weights to INT32 for 4x compression
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.precision = precision
        self.calibration_data = calibration_data
        self.num_calib_samples = num_calib_samples
        self.seq_len = seq_len
        self.device = device
        self.exclude_layers = exclude_layers or ["lm_head"]
        self.trust_remote_code = trust_remote_code
        self.sensitivity_threshold = sensitivity_threshold
        self.batch_size = batch_size
        self.pack_int4 = pack_int4 and precision == "int4"

        self.config = None
        self.tokenizer = None
        self.model_type = None
        self.template = None
        self.safetensors_files = []
        self.weight_index = {}  # Maps weight name to file path
        self.timing = {}
        self.sensitivity_analysis = {}

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _clear_memory(self) -> None:
        """Clear GPU and CPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _get_memory_info(self) -> str:
        """Get memory usage info."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU: {allocated:.2f}GB / {total:.2f}GB (reserved: {reserved:.2f}GB)"

        import psutil

        ram = psutil.virtual_memory()
        return f"RAM: {ram.used / 1024**3:.2f}GB / {ram.total / 1024**3:.2f}GB"

    def setup(self) -> None:
        """Load config, tokenizer, and build weight index."""
        start_time = time.time()
        self._log("Setting up lazy layer-wise quantization...")

        # Load config only (no weights)
        self.config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        self.model_type = getattr(self.config, "model_type", "unknown")
        self._log(f"Model type: {self.model_type}")

        # Get template
        available_templates = LLMTemplate.list_available()
        for template_name in available_templates:
            if self.model_type.lower() in template_name.lower():
                self.template = LLMTemplate.get(template_name)
                self._log(f"Using template: {template_name}")
                break

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )

        # Build weight index from safetensors files
        self._build_weight_index()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "quantized_layers"), exist_ok=True)

        self.timing["setup"] = time.time() - start_time
        self._log(f"Setup completed in {self.timing['setup']:.2f}s")

    def _build_weight_index(self) -> None:
        """Build index mapping weight names to safetensors files."""
        self._log("Building weight index from safetensors files...")

        model_path = Path(self.model_path)

        # Find all safetensors files
        self.safetensors_files = list(model_path.glob("*.safetensors"))
        if not self.safetensors_files:
            # Check for model.safetensors.index.json
            index_file = model_path / "model.safetensors.index.json"
            if index_file.exists():
                with open(index_file) as f:
                    index_data = json.load(f)
                self.weight_index = index_data.get("weight_map", {})
                self.safetensors_files = [model_path / f for f in set(self.weight_index.values())]
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

    def get_layer_info(self) -> dict[str, Any]:
        """Get layer information from config."""
        info = {
            "num_layers": getattr(self.config, "num_hidden_layers", 0),
            "layer_prefix": "model.layers",
            "hidden_size": getattr(self.config, "hidden_size", 0),
        }

        model_type_lower = self.model_type.lower()
        if (
            "llama" in model_type_lower
            or "qwen" in model_type_lower
            or "mistral" in model_type_lower
        ):
            info["layer_prefix"] = "model.layers"

        return info

    def _get_layer_weight_names(self, layer_idx: int, layer_prefix: str) -> list[str]:
        """Get all weight names for a specific layer."""
        layer_pattern = f"{layer_prefix}.{layer_idx}."
        return [k for k in self.weight_index.keys() if k.startswith(layer_pattern)]

    def _load_layer_weights(self, weight_names: list[str]) -> dict[str, torch.Tensor]:
        """Load weights for specific layer from safetensors."""
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

    def _create_layer_module(self, layer_idx: int, layer_prefix: str):
        """Create a single layer module with weights loaded."""
        # Import model class

        # Load just the config
        layer_weight_names = self._get_layer_weight_names(layer_idx, layer_prefix)

        if not layer_weight_names:
            self._log(f"Warning: No weights found for layer {layer_idx}")
            return None

        # We need to create a minimal model structure
        # For now, load full model structure but keep on CPU
        # This is a limitation - ideally we'd build the layer from scratch

        # Alternative: Load the layer weights directly and create a simple wrapper
        weights = self._load_layer_weights(layer_weight_names)

        # Determine layer type from weights
        has_attention = any("self_attn" in k for k in weights.keys())
        has_mlp = any("mlp" in k for k in weights.keys())

        self._log(f"  Layer {layer_idx} has {len(weights)} weights")
        self._log(f"  Attention: {has_attention}, MLP: {has_mlp}")

        return weights

    def quantize_layer_weights(
        self,
        layer_weights: dict[str, torch.Tensor],
        layer_idx: int,
        quant_config: QConfig,
    ) -> dict[str, Any]:
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
                        # Create Linear layer with transposed weight
                        linear = nn.Linear(in_features, out_features, bias=False)
                        linear.weight.data = tensor
                        # Use sanitized name for ModuleDict
                        safe_name = name.replace(".", "_")
                        self.linear_layers[safe_name] = linear

            def forward(self, x):
                return x  # Dummy forward

        # Create the layer with proper Linear modules
        linear_layer = LinearLayer(layer_weights)
        linear_layer = linear_layer.to(self.device)

        # Create quantizer
        quantizer = ModelQuantizer(quant_config)

        # Create dummy dataloader for weight-only quantization
        from torch.utils.data import DataLoader, TensorDataset

        dummy = torch.zeros(1, 1, device=self.device)
        dummy_loader = DataLoader(TensorDataset(dummy), batch_size=1)

        # Quantize
        quantized_layer = quantizer.quantize_model(linear_layer, dummy_loader)
        quantized_layer = quantizer.freeze(quantized_layer)

        # Extract quantized weights with original names
        # Map: linear_layers.{safe_name}.{suffix} -> {original_name}.{suffix}
        quantized_weights = {}
        safe_to_orig = {
            name.replace(".", "_"): name
            for name in layer_weights
            if len(layer_weights[name].shape) == 2
        }

        for name, param in quantized_layer.named_parameters():
            # Convert linear_layers.{safe_name}.weight -> {original_name}.weight
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
            # Convert linear_layers.{safe_name}._weight_quantizer.{param} -> {original_name}_quantizer.{param}
            if name.startswith("linear_layers.") and "_weight_quantizer" in name:
                parts = name.split(".", 2)
                if len(parts) == 3:
                    safe_name = parts[1]
                    quant_suffix = parts[2]  # e.g., "_weight_quantizer.scale"
                    if safe_name in safe_to_orig:
                        orig_name = safe_to_orig[safe_name]
                        # Store quantizer params with original weight name
                        new_name = f"{orig_name}{quant_suffix.replace('_weight', '')}"
                        quantized_weights[new_name] = buffer.data.cpu()
                    else:
                        quantized_weights[name] = buffer.data.cpu()
                else:
                    quantized_weights[name] = buffer.data.cpu()
            else:
                quantized_weights[name] = buffer.data.cpu()

        # Also copy non-2D tensors (like norms) as-is
        for orig_name, tensor in layer_weights.items():
            if len(tensor.shape) != 2:
                quantized_weights[orig_name] = tensor.cpu()

        # Clean up
        del linear_layer, quantized_layer, quantizer
        self._clear_memory()

        # Pack INT4 weights if requested
        if self.pack_int4:
            quantized_weights = self._pack_int4_weights(quantized_weights)

        return {
            "layer_idx": layer_idx,
            "weights": quantized_weights,
            "num_weights": len(quantized_weights),
        }

    def _pack_int4_weights(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pack quantized INT4 weights to INT32 format for storage efficiency."""
        packed_weights = {}

        # Find Linear weights (2D tensors with quantizer params)
        weight_keys = [
            k
            for k in weights
            if k.endswith(".weight") and "_quantizer" not in k and len(weights[k].shape) == 2
        ]

        for weight_key in weight_keys:
            scale_key = f"{weight_key}_quantizer.scale"
            zp_key = f"{weight_key}_quantizer.zero_point"

            if scale_key not in weights:
                # Not quantized, copy as-is
                packed_weights[weight_key] = weights[weight_key]
                continue

            weight = weights[weight_key]
            scale = weights[scale_key]
            zero_point = weights[zp_key]

            # Quantize to INT4 values
            int4_vals = quantize_to_int4(weight, scale, zero_point)

            # Pack to INT32
            packed = pack_int4_to_int32(int4_vals)

            # Save with packed format
            packed_weights[f"{weight_key}.packed"] = packed
            packed_weights[f"{weight_key}.scale"] = scale
            packed_weights[f"{weight_key}.zero_point"] = zero_point

        # Copy non-Linear weights (norms, etc.) and other tensors
        for key, tensor in weights.items():
            if "_quantizer" not in key and key not in weight_keys:
                packed_weights[key] = tensor

        return packed_weights

    def quantize_batch_layers(
        self,
        batch_weights: list[dict[str, torch.Tensor]],
        layer_indices: list[int],
        quant_config: QConfig,
    ) -> list[dict[str, Any]]:
        """Quantize multiple layers in a single pass for better GPU utilization."""

        class MultiLayerModule(nn.Module):
            """Module holding multiple layers for batch quantization."""

            def __init__(
                self, batch_weights: list[dict[str, torch.Tensor]], layer_indices: list[int]
            ):
                super().__init__()
                self.layers = nn.ModuleList()
                self.layer_names = []

                for layer_idx, weights in zip(layer_indices, batch_weights):
                    layer_dict = nn.ModuleDict()
                    for name, tensor in weights.items():
                        if len(tensor.shape) == 2:
                            out_features, in_features = tensor.shape
                            linear = nn.Linear(in_features, out_features, bias=False)
                            linear.weight.data = tensor
                            safe_name = f"layer{layer_idx}_{name.replace('.', '_')}"
                            layer_dict[safe_name] = linear
                            self.layer_names.append((layer_idx, name, safe_name))
                    self.layers.append(layer_dict)

            def forward(self, x):
                return x

        # Create multi-layer module
        multi_layer = MultiLayerModule(batch_weights, layer_indices)
        multi_layer = multi_layer.to(self.device)

        # Create quantizer
        quantizer = ModelQuantizer(quant_config)

        # Create dummy dataloader
        from torch.utils.data import DataLoader, TensorDataset

        dummy = torch.zeros(1, 1, device=self.device)
        dummy_loader = DataLoader(TensorDataset(dummy), batch_size=1)

        # Quantize all layers at once
        quantized_module = quantizer.quantize_model(multi_layer, dummy_loader)
        quantized_module = quantizer.freeze(quantized_module)

        # Extract weights for each layer
        results = []
        for batch_idx, (layer_idx, orig_weights) in enumerate(zip(layer_indices, batch_weights)):
            quantized_weights = {}

            # Build mapping for this layer
            safe_names_for_layer = [
                (orig_name, safe_name)
                for li, orig_name, safe_name in multi_layer.layer_names
                if li == layer_idx
            ]

            # Extract weights
            prefix = f"layers.{batch_idx}."
            for name, param in quantized_module.named_parameters():
                if name.startswith(prefix):
                    # Find original name
                    for orig_name, safe_name in safe_names_for_layer:
                        if safe_name in name:
                            new_name = name.replace(f"{prefix}{safe_name}.", "")
                            if new_name == "weight":
                                quantized_weights[orig_name] = param.data.cpu()
                            break

            for name, buffer in quantized_module.named_buffers():
                if name.startswith(prefix) and "_weight_quantizer" in name:
                    for orig_name, safe_name in safe_names_for_layer:
                        if safe_name in name:
                            quant_suffix = name.split("_weight_quantizer")[-1]
                            new_name = f"{orig_name}_quantizer{quant_suffix}"
                            quantized_weights[new_name] = buffer.data.cpu()
                            break

            # Copy non-2D tensors
            for orig_name, tensor in orig_weights.items():
                if len(tensor.shape) != 2:
                    quantized_weights[orig_name] = tensor.cpu()

            # Pack INT4 weights if requested
            if self.pack_int4:
                quantized_weights = self._pack_int4_weights(quantized_weights)

            results.append(
                {
                    "layer_idx": layer_idx,
                    "weights": quantized_weights,
                    "num_weights": len(quantized_weights),
                }
            )

        # Clean up
        del multi_layer, quantized_module, quantizer
        self._clear_memory()

        return results

    def run_quantization(self) -> dict[str, Any]:
        """Run the full layer-wise quantization pipeline."""
        total_start = time.time()

        self.setup()

        layer_info = self.get_layer_info()
        num_layers = layer_info["num_layers"]
        layer_prefix = layer_info["layer_prefix"]

        self._log(f"\n{'=' * 60}")
        self._log("Layer-wise Quantization Pipeline")
        self._log(f"{'=' * 60}")
        self._log(f"Model: {self.model_path}")
        self._log(f"Output: {self.output_dir}")
        self._log(f"Precision: {self.precision}")
        self._log(f"Pack INT4: {self.pack_int4}")
        self._log(f"Layers: {num_layers}")
        self._log(f"Batch size: {self.batch_size}")
        self._log(f"Device: {self.device}")
        self._log(f"{'=' * 60}")

        # Get quantization config
        quant_scheme = self.PRECISION_TO_SCHEME.get(self.precision, "int4_wo_128")
        self._log(f"Using quantization scheme: {quant_scheme}")

        if self.template:
            quant_config = self.template.get_config(
                scheme=quant_scheme,
                exclude_layers=self.exclude_layers,
            )
        else:
            from quark.torch.quantization.config.config import Int4PerGroupSpec

            quant_config = QConfig(
                global_quant_config=QLayerConfig(
                    weight=Int4PerGroupSpec(group_size=128).to_quantization_spec()
                ),
                exclude=self.exclude_layers,
            )

        # Process layers in batches for better GPU utilization
        quantized_layers = []
        layer_times = []

        self._log(f"Batch size: {self.batch_size}")
        total_batches = (num_layers + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(total_batches), desc="Quantizing batches"):
            batch_start = time.time()

            # Determine layer range for this batch
            start_layer = batch_idx * self.batch_size
            end_layer = min(start_layer + self.batch_size, num_layers)
            layer_indices = list(range(start_layer, end_layer))

            self._log(
                f"\n--- Batch {batch_idx + 1}/{total_batches} (Layers {start_layer}-{end_layer - 1}) ---"
            )
            self._clear_memory()
            self._log(f"  Memory: {self._get_memory_info()}")

            # Load weights for all layers in batch
            batch_weights = []
            for layer_idx in layer_indices:
                layer_weights = self._create_layer_module(layer_idx, layer_prefix)
                if layer_weights is not None:
                    batch_weights.append(layer_weights)
                else:
                    # Add empty dict for missing layers
                    batch_weights.append({})

            # Filter out empty batches
            valid_indices = [i for i, w in enumerate(batch_weights) if w]
            if not valid_indices:
                self._log("  Skipping empty batch")
                continue

            # Quantize batch
            if len(valid_indices) == 1 or self.batch_size == 1:
                # Single layer mode (original behavior)
                for i in valid_indices:
                    result = self.quantize_layer_weights(
                        batch_weights[i], layer_indices[i], quant_config
                    )
                    layer_file = (
                        Path(self.output_dir)
                        / "quantized_layers"
                        / f"layer_{layer_indices[i]}.safetensors"
                    )
                    save_file(result["weights"], layer_file)
                    quantized_layers.append(
                        {
                            "layer_idx": layer_indices[i],
                            "file": str(layer_file),
                            "num_weights": result["num_weights"],
                        }
                    )
            else:
                # Batch mode
                valid_batch_weights = [batch_weights[i] for i in valid_indices]
                valid_layer_indices = [layer_indices[i] for i in valid_indices]

                results = self.quantize_batch_layers(
                    valid_batch_weights, valid_layer_indices, quant_config
                )

                for result in results:
                    layer_file = (
                        Path(self.output_dir)
                        / "quantized_layers"
                        / f"layer_{result['layer_idx']}.safetensors"
                    )
                    save_file(result["weights"], layer_file)
                    quantized_layers.append(
                        {
                            "layer_idx": result["layer_idx"],
                            "file": str(layer_file),
                            "num_weights": result["num_weights"],
                        }
                    )

            batch_time = time.time() - batch_start
            layers_in_batch = len(valid_indices)
            layer_times.extend([batch_time / layers_in_batch] * layers_in_batch)
            avg_time = sum(layer_times) / len(layer_times)
            remaining_layers = num_layers - end_layer
            eta = avg_time * remaining_layers

            self._log(
                f"  Completed in {batch_time:.2f}s ({batch_time / layers_in_batch:.2f}s/layer, ETA: {eta / 60:.1f}m)"
            )

            # Clear memory
            del batch_weights
            self._clear_memory()

        # Save final metadata
        result = {
            "success": True,
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "model_type": self.model_type,
            "quant_scheme": quant_scheme,
            "precision": self.precision,
            "pack_int4": self.pack_int4,
            "batch_size": self.batch_size,
            "num_layers": num_layers,
            "quantized_layers": quantized_layers,
            "exclude_layers": self.exclude_layers,
            "timing": {
                "total": time.time() - total_start,
                "setup": self.timing.get("setup", 0),
                "avg_layer": sum(layer_times) / len(layer_times) if layer_times else 0,
            },
        }

        # Save result
        with open(Path(self.output_dir) / "quantization_result.json", "w") as f:
            json.dump(result, f, indent=2)

        self._log(f"\n{'=' * 60}")
        self._log("QUANTIZATION COMPLETE")
        self._log(f"{'=' * 60}")
        self._log(f"Total time: {result['timing']['total']:.2f}s")
        self._log(f"Layers quantized: {len(quantized_layers)}")
        self._log(f"Output: {self.output_dir}")
        self._log(f"{'=' * 60}")

        return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lazy Layer-wise Quantization")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--precision", default="int4", choices=["int4", "int8", "fp8"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude_layers", nargs="*", default=["lm_head"])
    parser.add_argument("--num_calib_samples", type=int, default=16)
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Number of layers to process in parallel"
    )
    parser.add_argument(
        "--pack_int4",
        action="store_true",
        default=True,
        help="Pack INT4 weights to INT32 (default: True)",
    )
    parser.add_argument("--no_pack_int4", action="store_true", help="Disable INT4 packing")

    args = parser.parse_args()

    quantizer = LazyLayerwiseQuantizer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        precision=args.precision,
        device=args.device,
        exclude_layers=args.exclude_layers,
        num_calib_samples=args.num_calib_samples,
        batch_size=args.batch_size,
        pack_int4=not args.no_pack_int4,
    )

    result = quantizer.run_quantization()
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())

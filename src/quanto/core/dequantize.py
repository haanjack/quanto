"""
Quanto: Model Dequantization Module

This module provides functionality to dequantize Quark-quantized models
back to floating point (BF16/FP16) format.

Dequantization formula: dequantized = (quantized - zero_point) * scale

Supports layer-wise dequantization for large models that don't fit in GPU memory.
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
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class DequantizationConfig:
    """Configuration for dequantization."""

    model_path: str
    output_dir: str
    output_dtype: str = "bf16"  # bf16, fp16, fp32
    device: str = "cuda"
    trust_remote_code: bool = True
    layerwise: bool = False  # Process one layer at a time for memory efficiency

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "output_dtype": self.output_dtype,
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "layerwise": self.layerwise,
        }


@dataclass
class DequantizationResult:
    """Result of dequantization process."""

    success: bool
    output_dir: str | None = None
    original_dtype: str | None = None
    output_dtype: str | None = None
    model_type: str | None = None
    quant_scheme: str | None = None
    error_message: str | None = None
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "original_dtype": self.original_dtype,
            "output_dtype": self.output_dtype,
            "model_type": self.model_type,
            "quant_scheme": self.quant_scheme,
            "error_message": self.error_message,
            "timing": self.timing,
        }


def unpack_int4_to_int8(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack INT4 values packed into INT32 tensor using Quark order.

    Quark packs 8 INT4 values into one INT32 using order [0,2,4,6,1,3,5,7].
    Format: [in_features, out_features//8] packed INT32
    Output: [in_features, out_features] INT8
    """
    in_features, out_features_packed = packed.shape
    out_features = out_features_packed * 8

    # Unpack using Quark order
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    unpacked = torch.zeros((in_features, out_features), dtype=torch.int8, device=packed.device)

    for i, idx in enumerate(order):
        shift = i * 4
        # Extract 4 bits
        nibble = ((packed >> shift) & 0xF).to(torch.int8)
        # Convert from unsigned to signed (handles negative values)
        nibble = torch.where(nibble >= 8, nibble - 16, nibble)
        unpacked[:, idx::8] = nibble

    return unpacked


def unpack_int8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack INT8 values (no packing needed, just cast)."""
    return packed.to(torch.int8)


def dequantize_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    weight_dtype: str = "int4",
) -> torch.Tensor:
    """
    Dequantize a weight tensor from packed INT4 format.

    Packed format from lazy_layerwise_quant:
    - Weight packed: [in_features, out_features // 8] INT32
    - Scale: [out_features, num_groups] BF16
    - Zero point: [out_features, num_groups] INT32

    Output: [out_features, in_features] dequantized tensor
    """
    # Unpack INT4 to INT8
    if weight_dtype == "int4":
        # weight is [in_features, out_features // 8] packed INT32
        # After unpack: [in_features, out_features] INT8
        weight_unpacked = unpack_int4_to_int8(weight)
    else:
        weight_unpacked = weight.to(torch.int8)

    in_features, out_features = weight_unpacked.shape

    # Transpose to [out_features, in_features]
    weight_unpacked = weight_unpacked.T.contiguous()

    # Scale shape: [out_features, num_groups]
    num_groups = scale.shape[1]
    group_size = in_features // num_groups

    # Reshape for per-group dequantization
    # weight: [out_features, in_features] -> [out_features, num_groups, group_size]
    weight_grouped = weight_unpacked.view(out_features, num_groups, group_size)

    # Scale and zero_point: [out_features, num_groups] -> [out_features, num_groups, 1]
    scale_exp = scale.unsqueeze(-1).float()
    zp_exp = zero_point.unsqueeze(-1).float()

    # Dequantize: (q - zp) * scale
    dequant = (weight_grouped.float() - zp_exp) * scale_exp

    # Reshape back to [out_features, in_features]
    dequant = dequant.view(out_features, in_features)

    return dequant


class ModelDequantizer:
    """
    Dequantize Quark-quantized models back to floating point.

    Supports layer-wise dequantization for large models.
    """

    DTYPE_MAP = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    def __init__(self, config: DequantizationConfig):
        self.config = config
        self.timing: dict[str, float] = {}
        self.model_config: dict[str, Any] = {}
        self.quant_config: dict[str, Any] = {}

    def _log(self, message: str) -> None:
        """Print log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _load_config(self) -> None:
        """Load model and quantization config."""
        config_path = Path(self.config.model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.model_config = json.load(f)
            self.quant_config = self.model_config.get("quantization_config", {})
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")

    def _get_weight_dtype(self) -> str:
        """Get the quantized weight dtype from config."""
        global_config = self.quant_config.get("global_quant_config", {})
        weight_config = global_config.get("weight", {})
        return weight_config.get("dtype", "int4")

    def _get_output_dtype(self) -> torch.dtype:
        """Get output dtype."""
        return self.DTYPE_MAP.get(self.config.output_dtype.lower(), torch.bfloat16)

    def dequantize(self) -> DequantizationResult:
        """
        Dequantize the model.

        Supports two formats:
        1. Layer files: quantized_layers/layer_*.safetensors (from lazy_layerwise_quant)
        2. Single model: model.safetensors

        Returns:
            DequantizationResult with details
        """
        result = DequantizationResult(success=False)

        try:
            start_time = time.time()

            model_path = Path(self.config.model_path)

            # Detect input format - check for layer files first
            layer_dir = model_path / "quantized_layers"
            if layer_dir.exists() and list(layer_dir.glob("layer_*.safetensors")):
                self._log(f"Detected layer-wise format from {layer_dir}")
                return self._dequantize_layer_files(layer_dir, result, start_time)

            # Load config for single model format
            self._log(f"Loading config from {self.config.model_path}...")
            self._load_config()

            model_type = self.model_config.get("model_type", "unknown")
            quant_dtype = self._get_weight_dtype()
            output_dtype = self._get_output_dtype()

            self._log(f"Model type: {model_type}")
            self._log(f"Quantization dtype: {quant_dtype}")
            self._log(f"Output dtype: {self.config.output_dtype}")

            result.original_dtype = quant_dtype
            result.output_dtype = self.config.output_dtype
            result.model_type = model_type
            result.quant_scheme = (
                self.quant_config.get("global_quant_config", {})
                .get("weight", {})
                .get("qscheme", "unknown")
            )

            # Check model file
            model_file = model_path / "model.safetensors"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")

            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)

            if self.config.layerwise:
                # Layer-wise dequantization for large models
                self._dequantize_layerwise(model_file, quant_dtype, output_dtype)
            else:
                # Standard in-memory dequantization
                self._dequantize_inmemory(model_file, quant_dtype, output_dtype)

            # Save config (remove quantization config)
            output_config = self.model_config.copy()
            if "quantization_config" in output_config:
                del output_config["quantization_config"]

            with open(os.path.join(self.config.output_dir, "config.json"), "w") as f:
                json.dump(output_config, f, indent=2)

            # Copy tokenizer files
            self._log("Copying tokenizer files...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=self.config.trust_remote_code
            )
            tokenizer.save_pretrained(self.config.output_dir)

            # Copy other files
            for fname in ["generation_config.json", "chat_template.jinja", "tokenizer.json"]:
                src = Path(self.config.model_path) / fname
                if src.exists():
                    import shutil

                    shutil.copy(src, Path(self.config.output_dir) / fname)

            self.timing["total"] = time.time() - start_time
            result.timing = self.timing
            result.success = True
            result.output_dir = self.config.output_dir

            self._print_summary(result)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._log(f"Error during dequantization: {e}")
            import traceback

            traceback.print_exc()

        return result

    def _dequantize_layer_files(
        self,
        layer_dir: Path,
        result: DequantizationResult,
        start_time: float,
    ) -> DequantizationResult:
        """
        Dequantize from layer files format (quantized_layers/layer_*.safetensors).

        This processes each layer file independently to minimize memory usage.
        """
        import gc

        self._log("Dequantizing from layer files format...")

        # Find all layer files
        layer_files = sorted(
            layer_dir.glob("layer_*.safetensors"), key=lambda x: int(x.stem.split("_")[1])
        )

        if not layer_files:
            raise FileNotFoundError(f"No layer files found in {layer_dir}")

        self._log(f"Found {len(layer_files)} layer files")

        # Load quantization result for metadata
        result_file = layer_dir.parent / "quantization_result.json"
        if result_file.exists():
            with open(result_file) as f:
                quant_result = json.load(f)
            result.model_type = quant_result.get("model_type", "unknown")
            result.original_dtype = quant_result.get("precision", "int4")
        else:
            result.model_type = "unknown"
            result.original_dtype = "int4"

        output_dtype = self._get_output_dtype()
        result.output_dtype = self.config.output_dtype

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_layer_dir = Path(self.config.output_dir) / "dequantized_layers"
        output_layer_dir.mkdir(exist_ok=True)

        # Process each layer file
        total_original = 0
        total_dequant = 0

        for layer_file in tqdm(layer_files, desc="Dequantizing layers"):
            # Load layer
            layer_weights = {}
            with safe_open(layer_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    layer_weights[key] = f.get_tensor(key)

            # Calculate original size
            original_size = sum(t.numel() * t.element_size() for t in layer_weights.values())
            total_original += original_size

            # Dequantize this layer
            dequant_weights = self._dequantize_layer(layer_weights, output_dtype)

            # Calculate dequantized size
            dequant_size = sum(t.numel() * t.element_size() for t in dequant_weights.values())
            total_dequant += dequant_size

            # Save immediately
            output_file = output_layer_dir / layer_file.name
            save_file(dequant_weights, str(output_file))

            # Clear memory
            del layer_weights, dequant_weights
            gc.collect()

        # Copy metadata files
        if result_file.exists():
            import shutil

            shutil.copy(result_file, Path(self.config.output_dir) / "quantization_result.json")

        # Try to copy config and tokenizer from original model path
        if result_file.exists():
            with open(result_file) as f:
                quant_result = json.load(f)
            original_model_path = quant_result.get("model_path")
            if original_model_path and Path(original_model_path).exists():
                self._copy_model_files(Path(original_model_path), Path(self.config.output_dir))

        self.timing["total"] = time.time() - start_time
        result.timing = self.timing
        result.success = True
        result.output_dir = self.config.output_dir

        # Print summary
        print(f"\n{'=' * 60}")
        print("DEQUANTIZATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Input: {layer_dir}")
        print(f"Output: {output_layer_dir}")
        print(f"Original size: {total_original / 1024**3:.2f} GB")
        print(f"Dequantized size: {total_dequant / 1024**3:.2f} GB")
        print(f"Total time: {self.timing['total']:.2f}s")
        print(f"{'=' * 60}")

        return result

    def _dequantize_layer(
        self,
        layer_weights: dict[str, torch.Tensor],
        output_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Dequantize a single layer's weights."""
        dequant_weights = {}

        # Find all packed weights (have .packed, .scale, .zero_point)
        packed_keys = [k for k in layer_weights if k.endswith(".weight.packed")]

        for packed_key in packed_keys:
            # Get base name: model.layers.0.mlp.gate_proj.weight.packed -> model.layers.0.mlp.gate_proj.weight
            base_key = packed_key.rsplit(".packed", 1)[0]
            scale_key = f"{base_key}.scale"
            zp_key = f"{base_key}.zero_point"

            if scale_key not in layer_weights or zp_key not in layer_weights:
                continue

            packed = layer_weights[packed_key]
            scale = layer_weights[scale_key]
            zero_point = layer_weights[zp_key]

            # Dequantize
            dequant = dequantize_weight(packed, scale, zero_point, "int4")
            dequant_weights[base_key] = dequant.to(output_dtype)

        # Copy non-packed weights (norms, etc.)
        for key, tensor in layer_weights.items():
            if ".packed" not in key and ".scale" not in key and ".zero_point" not in key:
                dequant_weights[key] = tensor.to(output_dtype)

        return dequant_weights

    def _copy_model_files(self, src_dir: Path, dst_dir: Path) -> None:
        """Copy model config and tokenizer files."""
        files_to_copy = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
        ]
        import shutil

        for fname in files_to_copy:
            src = src_dir / fname
            if src.exists():
                shutil.copy(src, dst_dir / fname)

    def _dequantize_inmemory(
        self,
        model_file: Path,
        quant_dtype: str,
        output_dtype: torch.dtype,
    ) -> None:
        """Standard in-memory dequantization for smaller models."""
        self._log(f"Loading quantized tensors from {model_file}...")
        tensors = {}
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)

        # Dequantize tensors
        self._log("Dequantizing tensors...")
        dequantized = {}
        quantized_keys = set()

        # Find all quantized weights (have associated scale/zero_point)
        for key in tensors:
            if key.endswith(".weight_quantizer.scale"):
                weight_key = key.replace(".weight_quantizer.scale", ".weight")
                zp_key = key.replace(".weight_quantizer.scale", ".weight_quantizer.zero_point")

                if weight_key in tensors and zp_key in tensors:
                    quantized_keys.add(weight_key)
                    quantized_keys.add(key)
                    quantized_keys.add(zp_key)

                    weight = tensors[weight_key]
                    scale = tensors[key]
                    zero_point = tensors[zp_key]

                    self._log(f"Dequantizing {weight_key}...")
                    dequant_weight = dequantize_weight(weight, scale, zero_point, quant_dtype)
                    dequantized[weight_key] = dequant_weight.to(output_dtype)

        # Copy non-quantized tensors as-is
        for key in tensors:
            if key not in quantized_keys:
                dequantized[key] = tensors[key].to(output_dtype)

        # Save
        self._log(f"Saving dequantized model to {self.config.output_dir}...")
        save_file(dequantized, os.path.join(self.config.output_dir, "model.safetensors"))

        # Clear memory
        del tensors
        del dequantized

    def _dequantize_layerwise(
        self,
        model_file: Path,
        quant_dtype: str,
        output_dtype: torch.dtype,
    ) -> None:
        """
        Layer-wise dequantization for large models.

        Processes one tensor at a time to minimize memory usage.
        Each quantized weight is dequantized and saved immediately.
        """
        self._log("Using layer-wise dequantization for large model...")

        # First pass: identify all quantized weight keys
        self._log("Scanning model structure...")
        quantized_weights = []
        all_keys = []

        with safe_open(model_file, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
            for key in all_keys:
                if key.endswith(".weight_quantizer.scale"):
                    weight_key = key.replace(".weight_quantizer.scale", ".weight")
                    zp_key = key.replace(".weight_quantizer.scale", ".weight_quantizer.zero_point")
                    if weight_key in all_keys and zp_key in all_keys:
                        quantized_weights.append((weight_key, key, zp_key))

        self._log(f"Found {len(quantized_weights)} quantized weight tensors")

        # Process tensors in batches to manage memory
        dequantized = {}
        processed_keys = set()

        # Process quantized weights one at a time
        for weight_key, scale_key, zp_key in tqdm(quantized_weights, desc="Dequantizing"):
            with safe_open(model_file, framework="pt", device="cpu") as f:
                weight = f.get_tensor(weight_key)
                scale = f.get_tensor(scale_key)
                zero_point = f.get_tensor(zp_key)

            # Dequantize
            dequant_weight = dequantize_weight(weight, scale, zero_point, quant_dtype)
            dequantized[weight_key] = dequant_weight.to(output_dtype)
            processed_keys.add(weight_key)
            processed_keys.add(scale_key)
            processed_keys.add(zp_key)

            # Clear memory
            del weight, scale, zero_point, dequant_weight

        # Copy non-quantized tensors
        self._log("Copying non-quantized tensors...")
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in tqdm(all_keys, desc="Copying"):
                if key not in processed_keys:
                    tensor = f.get_tensor(key)
                    dequantized[key] = tensor.to(output_dtype)
                    del tensor

        # Save
        self._log(f"Saving dequantized model to {self.config.output_dir}...")
        save_file(dequantized, os.path.join(self.config.output_dir, "model.safetensors"))

        # Clear memory
        del dequantized
        self._log("Layer-wise dequantization complete!")

    def _print_summary(self, result: DequantizationResult) -> None:
        """Print dequantization summary."""
        print("\n" + "=" * 60)
        print("DEQUANTIZATION SUMMARY")
        print("=" * 60)
        print(f"Model Type: {result.model_type}")
        print(f"Original Quantization: {result.original_dtype}")
        print(f"Output Dtype: {result.output_dtype}")
        print(f"Output Directory: {result.output_dir}")

        if result.timing:
            print("\nTiming:")
            for stage, duration in result.timing.items():
                print(f"  {stage}: {duration:.2f}s")

        print("=" * 60)


def main():
    """Main entry point for dequantization."""
    parser = argparse.ArgumentParser(
        description="Quanto: Dequantize Quark-quantized models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dequantize INT4 model to BF16
  python -m quanto --dequantize --model_path ./quantized/qwen3-32b-int4 --output_dir ./dequantized/qwen3-32b-bf16

  # Dequantize to FP16 with layer-wise processing for large models
  python -m quanto --dequantize --model_path ./quantized/qwen3-32b-int4 --output_dir ./dequantized/qwen3-32b-fp16 --output_dtype fp16 --layerwise
        """,
    )

    parser.add_argument(
        "--dequantize",
        action="store_true",
        help="Run dequantization mode",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the quantized model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dequantized model",
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Output data type (default: bf16)",
    )
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="Use layer-wise dequantization for large models (processes one tensor at a time)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading models",
    )

    args = parser.parse_args()

    if not args.dequantize:
        parser.error("--dequantize flag is required for dequantization mode")

    config = DequantizationConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        output_dtype=args.output_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        layerwise=args.layerwise,
    )

    dequantizer = ModelDequantizer(config)
    result = dequantizer.dequantize()

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = Path(args.output_dir) / "dequantization_result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {result_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

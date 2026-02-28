#!/usr/bin/env python3
"""
HuggingFace Export for Quantized Models

Converts layer-wise quantized checkpoints to HuggingFace-compatible format:
- Sharded safetensors files (~5GB each)
- Proper config.json with quantization metadata
- model.safetensors.index.json for weight mapping
- All tokenizer and config files

Usage:
    python hf_export.py --quantized_dir /output/qwen3-32b-int4-packed \
                        --output_dir /output/qwen3-32b-int4-hf \
                        --original_model_path /models/qwen3-32b \
                        --shard_size_gb 5
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


class HuggingFaceExporter:
    """
    Exports quantized layer files to HuggingFace-compatible format.
    """

    def __init__(
        self,
        quantized_dir: str,
        output_dir: str,
        original_model_path: str | None = None,
        shard_size_gb: float = 5.0,
        trust_remote_code: bool = True,
    ):
        self.quantized_dir = Path(quantized_dir)
        self.output_dir = Path(output_dir)
        self.original_model_path = Path(original_model_path) if original_model_path else None
        self.shard_size_bytes = int(shard_size_gb * 1024**3)
        self.trust_remote_code = trust_remote_code
        self.quant_result = None
        self.model_config = None

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Get tensor size in bytes."""
        return tensor.numel() * tensor.element_size()

    def load_quantization_result(self) -> dict[str, Any]:
        """Load the quantization result JSON."""
        result_file = self.quantized_dir / "quantization_result.json"
        if result_file.exists():
            with open(result_file) as f:
                self.quant_result = json.load(f)
            return self.quant_result
        raise FileNotFoundError(f"Quantization result not found: {result_file}")

    def load_model_config(self) -> dict[str, Any]:
        """Load model config from original model path or quantization result."""
        # Try original model path first
        if self.original_model_path:
            config_file = self.original_model_path / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    self.model_config = json.load(f)
                return self.model_config

        # Try to get from quantization result
        if self.quant_result and "model_path" in self.quant_result:
            config_path = Path(self.quant_result["model_path"]) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    self.model_config = json.load(f)
                self.original_model_path = Path(self.quant_result["model_path"])
                return self.model_config

        # Create minimal config from quantization result
        self._log("Warning: Could not find config.json, creating minimal config")
        self._log("WARNING: The exported model may not load correctly without a complete config!")
        self._log(
            "         Please provide --original_model_path with a valid path to the original model."
        )

        # Try to infer basic config from quantization result
        num_layers = self.quant_result.get("num_layers", 32) if self.quant_result else 32
        self.model_config = {
            "model_type": self.quant_result.get("model_type", "unknown")
            if self.quant_result
            else "unknown",
            "architectures": ["Qwen2ForCausalLM"],  # Default assumption
            "num_hidden_layers": num_layers,
        }
        return self.model_config

    def export(self) -> dict[str, Any]:
        """
        Export quantized model to HuggingFace format.
        """
        start_time = time.time()
        self._log("Exporting to HuggingFace format...")
        self._log(f"Input: {self.quantized_dir}")
        self._log(f"Output: {self.output_dir}")
        self._log(f"Shard size: {self.shard_size_bytes / 1024**3:.1f} GB")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata (quantization result first, then config which may depend on it)
        self.load_quantization_result()
        try:
            self.load_model_config()
        except Exception as e:
            self._log(f"Warning: Could not load model config: {e}")

        # Get layer files
        layer_dir = self.quantized_dir / "quantized_layers"
        if not layer_dir.exists():
            layer_dir = self.quantized_dir / "dequantized_layers"

        layer_files = sorted(
            layer_dir.glob("layer_*.safetensors"), key=lambda x: int(x.stem.split("_")[1])
        )

        if not layer_files:
            raise FileNotFoundError(f"No layer files found in {layer_dir}")

        self._log(f"Found {len(layer_files)} layer files")

        # Collect all tensor metadata first (without loading data)
        self._log("Collecting tensor metadata...")
        tensor_info = self._collect_tensor_metadata(layer_files)

        # Also collect non-layer tensors from original model
        non_layer_info = self._collect_non_layer_metadata()
        tensor_info.update(non_layer_info)

        self._log(f"Total tensors: {len(tensor_info)}")

        # Calculate total size and plan shards
        total_size = sum(info["size"] for info in tensor_info.values())
        self._log(f"Total model size: {total_size / 1024**3:.2f} GB")

        shards = self._plan_shards(tensor_info)
        self._log(f"Planned {len(shards)} shards")

        # Write shards
        weight_map = {}
        for shard_idx, (shard_name, tensor_names) in enumerate(tqdm(shards, desc="Writing shards")):
            self._write_shard(
                shard_idx, shard_name, tensor_names, tensor_info, layer_files, weight_map
            )
            gc.collect()

        # Write index file
        self._write_index(weight_map, len(shards))

        # Write config with quantization metadata
        self._write_config()

        # Copy tokenizer files
        self._copy_tokenizer_files()

        # Write export result
        result = {
            "success": True,
            "quantized_dir": str(self.quantized_dir),
            "output_dir": str(self.output_dir),
            "num_layers": len(layer_files),
            "num_tensors": len(tensor_info),
            "num_shards": len(shards),
            "total_size_gb": total_size / 1024**3,
            "shard_size_gb": self.shard_size_bytes / 1024**3,
            "timing": {"total": time.time() - start_time},
        }

        with open(self.output_dir / "export_result.json", "w") as f:
            json.dump(result, f, indent=2)

        self._log(f"\n{'=' * 60}")
        self._log("EXPORT COMPLETE")
        self._log(f"{'=' * 60}")
        self._log(f"Output: {self.output_dir}")
        self._log(f"Shards: {len(shards)}")
        self._log(f"Total size: {total_size / 1024**3:.2f} GB")
        self._log(f"Time: {result['timing']['total']:.1f}s")
        self._log(f"{'=' * 60}")

        return result

    def _collect_tensor_metadata(self, layer_files: list[Path]) -> dict[str, dict]:
        """Collect tensor names and sizes from layer files."""
        tensor_info = {}

        for layer_file in layer_files:
            with safe_open(layer_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    metadata = f.metadata()
                    tensor_info[key] = {
                        "size": metadata[key]["data_offsets"][1] - metadata[key]["data_offsets"][0]
                        if metadata and key in metadata
                        else 0,
                        "file": str(layer_file),
                    }

        # Fallback: load tensors to get sizes if metadata not available
        if any(v["size"] == 0 for v in tensor_info.values()):
            self._log("Loading tensors to calculate sizes...")
            tensor_info = {}
            for layer_file in tqdm(layer_files, desc="Scanning tensors"):
                with safe_open(layer_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        tensor_info[key] = {
                            "size": self._get_tensor_size(tensor),
                            "file": str(layer_file),
                        }
                        del tensor

        return tensor_info

    def _collect_non_layer_metadata(self) -> dict[str, dict]:
        """Collect metadata for non-layer tensors from original model."""
        non_layer_info = {}

        if not self.original_model_path:
            return non_layer_info

        # Check for index file
        index_file = self.original_model_path / "model.safetensors.index.json"
        weight_map_file = {}

        if index_file.exists():
            with open(index_file) as f:
                index_data = json.load(f)
            weight_map_file = index_data.get("weight_map", {})

        # Non-layer prefixes to include
        non_layer_prefixes = [
            "model.embed_tokens",
            "model.norm",
            "lm_head",
        ]

        # Find non-layer tensors
        for name, filename in weight_map_file.items():
            for prefix in non_layer_prefixes:
                if name.startswith(prefix):
                    # Get size from file
                    sf_path = self.original_model_path / filename
                    with safe_open(sf_path, framework="pt", device="cpu") as f:
                        if name in f.keys():
                            tensor = f.get_tensor(name)
                            non_layer_info[name] = {
                                "size": self._get_tensor_size(tensor),
                                "file": str(sf_path),
                                "is_non_layer": True,
                            }
                            del tensor
                    break

        # If no index file, scan all safetensors
        if not weight_map_file:
            for sf_file in self.original_model_path.glob("model*.safetensors"):
                with safe_open(sf_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        for prefix in non_layer_prefixes:
                            if key.startswith(prefix):
                                tensor = f.get_tensor(key)
                                non_layer_info[key] = {
                                    "size": self._get_tensor_size(tensor),
                                    "file": str(sf_file),
                                    "is_non_layer": True,
                                }
                                del tensor
                                break

        self._log(f"Found {len(non_layer_info)} non-layer tensors")
        return non_layer_info

    def _plan_shards(self, tensor_info: dict[str, dict]) -> list[tuple[str, list[str]]]:
        """Plan how to distribute tensors across shards."""
        shards = []
        current_shard = []
        current_size = 0
        shard_idx = 0

        # Sort tensors by name for consistent ordering
        sorted_tensors = sorted(tensor_info.items(), key=lambda x: x[0])

        for name, info in sorted_tensors:
            tensor_size = info["size"]

            # If single tensor is larger than shard size, it gets its own shard
            if tensor_size > self.shard_size_bytes:
                if current_shard:
                    shards.append((f"model-{shard_idx + 1:05d}", current_shard))
                    shard_idx += 1
                    current_shard = []
                    current_size = 0
                shards.append((f"model-{shard_idx + 1:05d}", [name]))
                shard_idx += 1
                continue

            # Check if adding this tensor would exceed shard size
            if current_size + tensor_size > self.shard_size_bytes and current_shard:
                shards.append((f"model-{shard_idx + 1:05d}", current_shard))
                shard_idx += 1
                current_shard = []
                current_size = 0

            current_shard.append(name)
            current_size += tensor_size

        # Don't forget the last shard
        if current_shard:
            shards.append((f"model-{shard_idx + 1:05d}", current_shard))

        return shards

    def _convert_tensor_name_to_quark_format(self, name: str) -> str:
        """Convert tensor names from internal format to HuggingFace/Quark format.

        For HuggingFace to load the model directly with from_pretrained(),
        we use the internal key format that Quark's modules expect:
        - `*.weight.packed` -> `*.weight`
        - `*.weight.scale` -> `*.weight_quantizer.scale`
        - `*.weight.zero_point` -> `*.weight_quantizer.zero_point`
        """
        if name.endswith(".weight.packed"):
            return name[:-7]  # Remove '.packed' -> '*.weight'
        elif name.endswith(".weight.scale"):
            # 'xxx.weight.scale' -> 'xxx.weight_quantizer.scale'
            return name[:-6] + "_quantizer.scale"
        elif name.endswith(".weight.zero_point"):
            # 'xxx.weight.zero_point' -> 'xxx.weight_quantizer.zero_point'
            return name[:-11] + "_quantizer.zero_point"
        return name

    def _needs_transpose(self, name: str) -> bool:
        """Check if a tensor needs to be transposed.

        Scale and zero_point tensors need to be transposed for INT4 per_group quantization.
        Quark expects scale shape: [num_groups, out_features] = [in_features // group_size, out_features]
        Our checkpoint has: [out_features, num_groups] = [out_features, in_features // group_size]
        """
        return name.endswith(".weight.scale") or name.endswith(".weight.zero_point")

    def _transpose_packed_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transpose scale/zero_point from our format to Quark's expected format."""
        return tensor.T.contiguous()

    def _write_shard(
        self,
        shard_idx: int,
        shard_name: str,
        tensor_names: list[str],
        tensor_info: dict[str, dict],
        layer_files: list[Path],
        weight_map: dict[str, str],
    ) -> None:
        """Write a single shard file."""
        shard_tensors = {}

        # Group by source file for efficient loading
        tensors_by_file = {}
        for name in tensor_names:
            info = tensor_info[name]
            src_file = info["file"]
            if src_file not in tensors_by_file:
                tensors_by_file[src_file] = []
            tensors_by_file[src_file].append(name)

        # Load tensors
        for src_file, names in tensors_by_file.items():
            with safe_open(src_file, framework="pt", device="cpu") as f:
                for name in names:
                    shard_tensors[name] = f.get_tensor(name)

        # Convert tensor names to Quark format and transpose packed weights
        quark_tensors = {}
        for name, tensor in shard_tensors.items():
            quark_name = self._convert_tensor_name_to_quark_format(name)

            # Transpose packed weights from Quark format to HF format
            if self._needs_transpose(name):
                tensor = self._transpose_packed_weight(tensor)

            quark_tensors[quark_name] = tensor

        # Write shard
        shard_filename = f"{shard_name}.safetensors"
        save_file(quark_tensors, str(self.output_dir / shard_filename))

        # Update weight map with converted names
        for name in tensor_names:
            quark_name = self._convert_tensor_name_to_quark_format(name)
            weight_map[quark_name] = shard_filename

        del shard_tensors
        del quark_tensors

    def _write_index(self, weight_map: dict[str, str], num_shards: int) -> None:
        """Write the model.safetensors.index.json file."""
        index = {
            "metadata": {
                "total_size": sum(
                    self._get_file_size(self.output_dir / f) for f in set(weight_map.values())
                ),
                "format": "pt",
            },
            "weight_map": weight_map,
        }

        with open(self.output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

        self._log(f"Written index with {len(weight_map)} tensors")

    def _get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        return path.stat().st_size

    def _write_config(self) -> None:
        """Write config.json with quantization metadata."""
        config = self.model_config.copy() if self.model_config else {}

        # Get quantization parameters from result
        quant_result = self.quant_result or {}
        quant_scheme = quant_result.get("quant_scheme", "int4_wo_128")
        precision = quant_result.get("precision", "int4")
        packed = quant_result.get("packed", True)
        exclude_layers = quant_result.get("exclude_layers", ["lm_head"])

        # Parse group size from scheme (e.g., "int4_wo_128" -> 128)
        group_size = 128
        if quant_scheme:
            parts = quant_scheme.split("_")
            if len(parts) >= 3:
                try:
                    group_size = int(parts[-1])
                except ValueError:
                    pass

        # Ensure required config fields are present
        if "model_type" not in config:
            config["model_type"] = quant_result.get("model_type", "unknown")
        if "architectures" not in config:
            config["architectures"] = ["Qwen2ForCausalLM"]
        if "torch_dtype" not in config:
            config["torch_dtype"] = "bfloat16"

        # Add quantization config for HuggingFace (matching AMD/Quark format)
        config["quantization_config"] = {
            "quant_method": "quark",
            "quant_mode": "eager_mode",
            "algo_config": None,
            "exclude": exclude_layers,
            "global_quant_config": {
                "weight": {
                    "ch_axis": -1,
                    "dtype": precision,
                    "group_size": group_size,
                    "is_dynamic": False,
                    "is_mx_scale_constraint": False,
                    "is_scale_quant": False,
                    "mx_element_dtype": None,
                    "observer_cls": "PerGroupMinMaxObserver",
                    "qscheme": "per_group",
                    "round_method": "half_even",
                    "scale_calculation_mode": "even",
                    "scale_format": "bf16",
                    "scale_type": "float",
                    "symmetric": False,
                },
                "bias": None,
                "input_tensors": None,
                "output_tensors": None,
                "target_device": None,
            },
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "softmax_quant_spec": None,
            "export": {
                "kv_cache_group": [],
                "min_kv_scale": 0.0,
                "pack_method": "reorder" if packed else None,
                "weight_format": "real_quantized" if packed else "fp",
                "weight_merge_groups": None,
            },
        }

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self._log("Written config.json with quantization metadata")

    def _copy_tokenizer_files(self) -> None:
        """Copy tokenizer and other metadata files."""
        if not self.original_model_path:
            self._log("Warning: No original model path, skipping tokenizer files")
            return

        if not self.original_model_path.exists():
            self._log(f"Warning: Original model path does not exist: {self.original_model_path}")
            return

        files_to_copy = [
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
            "generation_config.json",
            "chat_template.jinja",
            "preprocessor_config.json",
        ]

        copied = []
        for fname in files_to_copy:
            src = self.original_model_path / fname
            if src.exists():
                shutil.copy(src, self.output_dir / fname)
                copied.append(fname)

        if copied:
            self._log(f"Copied tokenizer files: {copied}")
        else:
            self._log("Warning: No tokenizer files found to copy")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export quantized model to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export packed INT4 model
    python hf_export.py --quantized_dir /output/qwen3-32b-int4-packed \\
                        --output_dir /output/qwen3-32b-int4-hf \\
                        --original_model_path /models/qwen3-32b

    # Export dequantized BF16 model with custom shard size
    python hf_export.py --quantized_dir /output/qwen3-32b-dequant-bf16 \\
                        --output_dir /output/qwen3-32b-bf16-hf \\
                        --original_model_path /models/qwen3-32b \\
                        --shard_size_gb 10
        """,
    )

    parser.add_argument(
        "--quantized_dir",
        required=True,
        help="Directory with quantized layers (quantized_layers/ or dequantized_layers/)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for HuggingFace format",
    )
    parser.add_argument(
        "--original_model_path",
        required=False,
        help="Path to original model for config, tokenizer, and non-layer weights. If not provided, will try to read from quantization result.",
    )
    parser.add_argument(
        "--shard_size_gb",
        type=float,
        default=5.0,
        help="Target shard size in GB (default: 5.0)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code",
    )

    args = parser.parse_args()

    exporter = HuggingFaceExporter(
        quantized_dir=args.quantized_dir,
        output_dir=args.output_dir,
        original_model_path=args.original_model_path,
        shard_size_gb=args.shard_size_gb,
        trust_remote_code=args.trust_remote_code,
    )

    result = exporter.export()

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

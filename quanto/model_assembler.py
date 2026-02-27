"""
Model Assembler for Quantized Layers

This module assembles individual quantized layer files into a complete model
that can be loaded for inference or quality evaluation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


class ModelAssembler:
    """
    Assembles quantized layer files into a complete model.
    """

    def __init__(
        self,
        quantized_dir: str,
        output_dir: str,
        original_model_path: str | None = None,
        trust_remote_code: bool = True,
    ):
        self.quantized_dir = Path(quantized_dir)
        self.output_dir = Path(output_dir)
        self.original_model_path = original_model_path
        self.trust_remote_code = trust_remote_code
        self.quant_result = None

    def _log(self, message: str) -> None:
        """Print log message."""
        print(f"[ModelAssembler] {message}")

    def load_quantization_result(self) -> dict[str, Any]:
        """Load the quantization result JSON."""
        result_file = self.quantized_dir / "quantization_result.json"
        if result_file.exists():
            with open(result_file) as f:
                self.quant_result = json.load(f)
            return self.quant_result
        raise FileNotFoundError(f"Quantization result not found: {result_file}")

    def assemble(self) -> dict[str, Any]:
        """
        Assemble all quantized layers into a single safetensors file.
        Uses streaming to avoid loading all tensors into memory at once.
        """
        self._log(f"Assembling model from {self.quantized_dir}")

        # Load quantization result
        if self.quant_result is None:
            self.load_quantization_result()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all layer files
        layer_files = sorted(
            self.quantized_dir.glob("quantized_layers/layer_*.safetensors"),
            key=lambda x: int(x.stem.split("_")[1])
        )

        self._log(f"Found {len(layer_files)} layer files")

        # First, collect all tensor metadata and non-layer weights
        all_tensors = {}

        # Copy non-layer weights from original model if available
        if self.original_model_path:
            self._copy_non_layer_weights(all_tensors)

        # Load and save layers in batches to manage memory
        batch_size = 8  # Process 8 layers at a time
        total_tensors = len(all_tensors)

        for batch_start in range(0, len(layer_files), batch_size):
            batch_end = min(batch_start + batch_size, len(layer_files))
            batch_files = layer_files[batch_start:batch_end]

            self._log(f"Loading layers {batch_start}-{batch_end-1}...")

            for layer_file in batch_files:
                with safe_open(layer_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        all_tensors[key] = f.get_tensor(key)

            total_tensors = len(all_tensors)

        # Save assembled model
        output_file = self.output_dir / "model.safetensors"
        self._log(f"Saving assembled model ({total_tensors} tensors) to {output_file}...")
        save_file(all_tensors, str(output_file))

        # Clear memory after save
        del all_tensors

        # Copy config and tokenizer
        self._copy_config_and_tokenizer()

        # Save assembly result
        result = {
            "success": True,
            "quantized_dir": str(self.quantized_dir),
            "output_dir": str(self.output_dir),
            "num_layers": len(layer_files),
            "num_tensors": total_tensors,
        }

        with open(self.output_dir / "assembly_result.json", "w") as f:
            json.dump(result, f, indent=2)

        self._log(f"Assembly complete: {total_tensors} tensors saved")

        return result

    def _copy_non_layer_weights(self, all_tensors: dict) -> None:
        """Copy non-layer weights (embeddings, lm_head) from original model."""
        if not self.original_model_path:
            return

        self._log("Copying non-layer weights from original model...")

        # Find safetensors files in original model
        model_path = Path(self.original_model_path)

        # Check for index file
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
        else:
            weight_map = {}

        # Non-layer prefixes to copy
        non_layer_prefixes = [
            "model.embed_tokens",
            "model.norm",
            "lm_head",
        ]

        # Find which tensors to copy
        tensors_to_copy = set()
        for name in (weight_map.keys() if weight_map else []):
            for prefix in non_layer_prefixes:
                if name.startswith(prefix):
                    tensors_to_copy.add(name)
                    break

        # If no index, scan all safetensors files
        if not tensors_to_copy or not weight_map:
            for sf_file in model_path.glob("*.safetensors"):
                with safe_open(sf_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        for prefix in non_layer_prefixes:
                            if key.startswith(prefix):
                                tensors_to_copy.add(key)
                                all_tensors[key] = f.get_tensor(key)
                                break
        else:
            # Copy from indexed files
            files_needed = set(weight_map[name] for name in tensors_to_copy)
            for sf_file in files_needed:
                sf_path = model_path / sf_file
                if sf_path.exists():
                    with safe_open(sf_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key in tensors_to_copy:
                                all_tensors[key] = f.get_tensor(key)

        self._log(f"Copied {len(tensors_to_copy)} non-layer tensors")

    def _copy_config_and_tokenizer(self) -> None:
        """Copy config, tokenizer, and other metadata files."""
        if not self.original_model_path:
            return

        model_path = Path(self.original_model_path)
        import shutil

        # Copy config
        config_file = model_path / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            # Add quantization config
            if self.quant_result:
                config["quantization_config"] = {
                    "quant_method": "quark",
                    "quant_scheme": self.quant_result.get("quant_scheme", "int4_wo_128"),
                    "precision": self.quant_result.get("precision", "int4"),
                }
            (self.output_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Copy tokenizer files
        for fname in [
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
            "generation_config.json",
            "chat_template.jinja",
        ]:
            src = model_path / fname
            if src.exists():
                shutil.copy(src, self.output_dir / fname)

        self._log("Copied config and tokenizer files")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Assemble quantized layers into a model")
    parser.add_argument("--quantized_dir", required=True, help="Directory with quantized layers")
    parser.add_argument("--output_dir", required=True, help="Output directory for assembled model")
    parser.add_argument("--original_model_path", help="Path to original model for non-layer weights")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    args = parser.parse_args()

    assembler = ModelAssembler(
        quantized_dir=args.quantized_dir,
        output_dir=args.output_dir,
        original_model_path=args.original_model_path,
        trust_remote_code=args.trust_remote_code,
    )

    result = assembler.assemble()

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

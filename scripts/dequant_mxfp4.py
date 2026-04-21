"""Dequantize MXFP4 packed expert weights back to BF16 for quality verification."""
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant_mxfp4_to_bf16(weight_uint8: torch.Tensor, scale_uint8: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP4 packed uint8 weights to bfloat16.

    Each uint8 byte holds 2 FP4 E2M1 values (nibble-packed).
    Scale is E8M0 format: value = 2^(e - 127), one per 32-element block.

    Args:
        weight_uint8: packed MXFP4 weights [..., K//2] as uint8
        scale_uint8: E8M0 block scales [..., K//32] as uint8
    Returns:
        dequantized weights [..., K] as bfloat16
    """
    # FP4 E2M1 lookup table: maps 4-bit value to float
    # E2M1: sign(1) exponent(2) mantissa(1)
    # Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6
    fp4_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )

    orig_shape = weight_uint8.shape  # [..., K//2]
    flat_w = weight_uint8.reshape(-1, weight_uint8.shape[-1])  # [N, K//2]
    flat_s = scale_uint8.reshape(-1, scale_uint8.shape[-1])  # [N, K//32]

    n_rows, k_half = flat_w.shape
    k = k_half * 2

    # Unpack nibbles: low nibble first, high nibble second (standard MXFP4 packing)
    lo = (flat_w & 0x0F).to(torch.int64)  # [N, K//2]
    hi = ((flat_w >> 4) & 0x0F).to(torch.int64)  # [N, K//2]

    # Interleave: [lo0, hi0, lo1, hi1, ...] → [N, K]
    unpacked = torch.stack([lo, hi], dim=-1).reshape(n_rows, k)  # [N, K]

    # Lookup FP4 values
    fp4_values = fp4_lut[unpacked]  # [N, K] float32

    # Decode E8M0 scales: value = 2^(e - 127)
    scale_float = (2.0 ** (flat_s.to(torch.float32) - 127.0))  # [N, K//32]

    # Each scale covers 32 elements (BLOCK_QUANT_DIM=32)
    # Repeat scale to match weight dimensions
    scale_expanded = scale_float.unsqueeze(-1).expand(-1, -1, 32).reshape(n_rows, -1)  # [N, K]
    # Trim if K is not exactly divisible
    scale_expanded = scale_expanded[:, :k]

    # Dequantize
    result = (fp4_values * scale_expanded).to(torch.bfloat16)

    # Reshape back to original shape but with last dim doubled
    new_shape = list(orig_shape)
    new_shape[-1] = k
    return result.reshape(new_shape)


def main():
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    os.makedirs(dst_dir, exist_ok=True)

    # Copy non-safetensors files
    for fname in os.listdir(src_dir):
        if not fname.endswith(".safetensors"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if os.path.isfile(src):
                import shutil

                shutil.copy2(src, dst)

    safetensors_files = sorted(glob.glob(os.path.join(src_dir, "model-*.safetensors")))
    print(f"Processing {len(safetensors_files)} safetensors files...")

    total_dequant = 0
    total_kept = 0

    for fpath in safetensors_files:
        fname = os.path.basename(fpath)
        print(f"\n  {fname}...")
        out_tensors = {}

        with safe_open(fpath, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for key in sorted(keys):
                # Skip scale tensors (handled with their weight)
                if key.endswith("_scale"):
                    continue

                tensor = f.get_tensor(key)
                scale_key = key + "_scale"

                if tensor.dtype == torch.uint8 and scale_key in keys:
                    # This is a quantized weight — dequantize
                    scale = f.get_tensor(scale_key)
                    dequantized = dequant_mxfp4_to_bf16(tensor, scale)
                    out_tensors[key] = dequantized
                    total_dequant += 1
                    print(f"    dequant: {key} {list(tensor.shape)} → {list(dequantized.shape)}")
                else:
                    out_tensors[key] = tensor
                    total_kept += 1

        out_path = os.path.join(dst_dir, fname)
        save_file(out_tensors, out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"    Saved {fname} ({size_mb:.1f} MB)")

    # Update config.json: remove quantization_config
    config_path = os.path.join(dst_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if "quantization_config" in config:
            del config["quantization_config"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print("\nRemoved quantization_config from config.json")

    # Rebuild safetensors index
    index_path = os.path.join(dst_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = {}
        total_size = 0
        for fpath in sorted(glob.glob(os.path.join(dst_dir, "model-*.safetensors"))):
            fname = os.path.basename(fpath)
            with safe_open(fpath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    weight_map[key] = fname
                    total_size += t.numel() * t.element_size()
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"Rebuilt index: {len(weight_map)} tensors, {total_size / 1e9:.2f} GB")

    print(f"\nDone: {total_dequant} tensors dequantized, {total_kept} kept as-is")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()

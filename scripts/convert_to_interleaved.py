"""
Convert HF-standard fused expert weights (contiguous gate|up blocks)
to GPT-OSS interleaved format (gate0,up0,gate1,up1,...) in-place.

vLLM's QuarkOCP_MX_MoEMethod uses convert_gpt_oss_weight_to_mxfp4_moe_kernel_format
which expects interleaved gate_up_proj rows. HuggingFace stores them as contiguous blocks.

This script converts:
  [gate_block (rows 0..N-1), up_block (rows N..2N-1)]
  →
  [gate0, up0, gate1, up1, ..., gate_N-1, up_N-1]

For both the packed weight and its scale tensor.
"""

import glob
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def contiguous_to_interleaved(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert [E, 2*N, K] from contiguous [gate_block|up_block]
    to interleaved [gate0, up0, gate1, up1, ...].

    Input layout:  [E, [gate_0..gate_{N-1}, up_0..up_{N-1}], K]
    Output layout: [E, [gate_0, up_0, gate_1, up_1, ...], K]
    """
    e, two_n, k = tensor.shape
    n = two_n // 2
    # Split into gate and up blocks: [E, 2, N, K]
    blocks = tensor.view(e, 2, n, k)
    # Permute to [E, N, 2, K] — pairs gate_i with up_i
    interleaved = blocks.permute(0, 2, 1, 3).contiguous()
    # Flatten back to [E, 2*N, K]
    return interleaved.view(e, two_n, k)


def main():
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    os.makedirs(dst_dir, exist_ok=True)

    # Copy non-safetensors files
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src) and not fname.endswith(".safetensors"):
            import shutil

            shutil.copy2(src, dst)

    safetensors_files = sorted(glob.glob(os.path.join(src_dir, "model-*.safetensors")))
    print(f"Processing {len(safetensors_files)} files...")

    converted = 0

    for fpath in safetensors_files:
        fname = os.path.basename(fpath)
        out_tensors = {}

        # load_file copies tensors into memory; safe_open tensors are invalid
        # outside the context manager (segfault on access).
        from safetensors.torch import load_file

        tensors = load_file(fpath)
        for key in sorted(tensors.keys()):
            tensor = tensors[key]

            # Convert gate_up_proj and its scale (3D fused expert tensors)
            if "gate_up_proj" in key and tensor.dim() == 3:
                out_tensors[key] = contiguous_to_interleaved(tensor)
                converted += 1
                print(f"  interleaved: {key} {list(tensor.shape)}")
            else:
                out_tensors[key] = tensor

        out_path = os.path.join(dst_dir, fname)
        save_file(out_tensors, out_path)

    # Rebuild index
    index_path = os.path.join(dst_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json

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

    print(f"\nDone: {converted} tensors interleaved")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()

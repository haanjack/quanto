"""
Unfuse 3D fused expert tensors into per-expert 2D tensors.

Converts HuggingFace fused format:
  experts.gate_up_proj [num_experts, 2*intermediate, hidden//2]  (packed MXFP4)
  experts.gate_up_proj_scale [num_experts, 2*intermediate, hidden//32]
  experts.down_proj [num_experts, hidden, intermediate//2]
  experts.down_proj_scale [num_experts, hidden, intermediate//32]

To per-expert format expected by vLLM's standard weight loader:
  experts.0.gate_proj.weight [intermediate, hidden//2]
  experts.0.gate_proj.weight_scale [intermediate, hidden//32]
  experts.0.up_proj.weight [intermediate, hidden//2]
  experts.0.up_proj.weight_scale [intermediate, hidden//32]
  experts.0.down_proj.weight [hidden, intermediate//2]
  experts.0.down_proj.weight_scale [hidden, intermediate//32]
  ...
"""

import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def unfuse_gate_up(tensor: torch.Tensor):
    """Split [E, 2*N, K] into gate [E, N, K] and up [E, N, K]."""
    e, two_n, k = tensor.shape
    n = two_n // 2
    gate = tensor[:, :n, :]  # First half = gate
    up = tensor[:, n:, :]    # Second half = up
    return gate, up


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

    total_unfused = 0
    total_new = 0

    for fpath in safetensors_files:
        fname = os.path.basename(fpath)
        out_tensors = {}

        with safe_open(fpath, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            for key in keys:
                tensor = f.get_tensor(key)

                if "experts.gate_up_proj" in key and tensor.dim() == 3:
                    # Unfuse gate_up_proj or gate_up_proj_scale
                    is_scale = key.endswith("_scale")
                    gate, up = unfuse_gate_up(tensor)
                    num_experts = tensor.shape[0]

                    base = key.rsplit("experts.gate_up_proj", 1)[0]
                    suffix = key.rsplit("experts.gate_up_proj", 1)[1]  # "" or "_scale"

                    for expert_id in range(num_experts):
                        if is_scale:
                            gate_name = f"{base}experts.{expert_id}.gate_proj.weight_scale"
                            up_name = f"{base}experts.{expert_id}.up_proj.weight_scale"
                        else:
                            gate_name = f"{base}experts.{expert_id}.gate_proj.weight"
                            up_name = f"{base}experts.{expert_id}.up_proj.weight"

                        out_tensors[gate_name] = gate[expert_id].contiguous()
                        out_tensors[up_name] = up[expert_id].contiguous()
                        total_new += 2

                    total_unfused += 1
                    print(f"  unfused: {key} [{num_experts} experts] → gate_proj + up_proj")

                elif "experts.down_proj" in key and tensor.dim() == 3:
                    # Unfuse down_proj or down_proj_scale
                    is_scale = key.endswith("_scale")
                    num_experts = tensor.shape[0]

                    base = key.rsplit("experts.down_proj", 1)[0]

                    for expert_id in range(num_experts):
                        if is_scale:
                            name = f"{base}experts.{expert_id}.down_proj.weight_scale"
                        else:
                            name = f"{base}experts.{expert_id}.down_proj.weight"

                        out_tensors[name] = tensor[expert_id].contiguous()
                        total_new += 1

                    total_unfused += 1
                    print(f"  unfused: {key} [{num_experts} experts] → down_proj")

                else:
                    out_tensors[key] = tensor

        out_path = os.path.join(dst_dir, fname)
        save_file(out_tensors, out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Saved {fname} ({size_mb:.1f} MB, {len(out_tensors)} tensors)")

    # Rebuild index
    index_path = os.path.join(dst_dir, "model.safetensors.index.json")
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

    print(f"\nDone: {total_unfused} fused tensors → {total_new} individual expert tensors")
    print(f"Index: {len(weight_map)} tensors, {total_size / 1e9:.2f} GB")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()

"""
Quanto: General Purpose LLM Quantization Tool
Memory management utilities.
"""

from __future__ import annotations

import gc

import torch


def clear_gpu_memory() -> None:
    """
    Clear GPU memory by running garbage collection and emptying CUDA cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_info() -> str:
    """
    Get current GPU memory usage information.

    Returns:
        String with memory allocation info in GB
    """
    if not torch.cuda.is_available():
        return "CUDA not available"

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    return (
        f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak"
    )


def get_device_memory_gb(device: torch.device | None = None) -> float:
    """
    Get total memory of device in GB.

    Args:
        device: Torch device (default: current CUDA device)

    Returns:
        Total memory in GB
    """
    if device is None:
        device = torch.device("cuda")

    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_properties(device).total_memory / 1024**3

    return 0.0


def get_free_memory_gb(device: torch.device | None = None) -> float:
    """
    Get free memory of device in GB.

    Args:
        device: Torch device (default: current CUDA device)

    Returns:
        Free memory in GB
    """
    if device is None:
        device = torch.device("cuda")

    if device.type == "cuda" and torch.cuda.is_available():
        total = get_device_memory_gb(device)
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        return total - allocated

    return 0.0


def print_memory_usage(label: str = "Memory") -> None:
    """
    Print current memory usage with a label.

    Args:
        label: Label for the print statement
    """
    info = get_memory_info()
    print(f"[{label}] {info}")


def estimate_model_memory_gb(
    num_params: int, dtype: torch.dtype = torch.bfloat16, include_overhead: bool = True
) -> float:
    """
    Estimate memory required to load a model.

    Args:
        num_params: Number of parameters
        dtype: Data type for weights
        include_overhead: Add 20% overhead for activations, gradients

    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 2)

    base_memory = num_params * bytes_per_param / 1024**3

    if include_overhead:
        return base_memory * 1.2

    return base_memory

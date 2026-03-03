"""
Activation Cache for Sequential Sensitivity Analysis.

Stores intermediate layer activations on GPU by default for fast access,
with optional CPU spillover when GPU memory is constrained.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from enum import Enum

import torch


class CacheLocation(Enum):
    """Storage location for cached activations."""
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class CacheStats:
    """Statistics for the activation cache."""
    num_entries: int = 0
    gpu_memory_bytes: int = 0
    cpu_memory_bytes: int = 0
    hits: int = 0
    misses: int = 0

    @property
    def gpu_memory_gb(self) -> float:
        return self.gpu_memory_bytes / (1024 ** 3)

    @property
    def cpu_memory_gb(self) -> float:
        return self.cpu_memory_bytes / (1024 ** 3)


@dataclass
class CachedActivation:
    """A single cached activation tensor."""
    tensor: torch.Tensor
    layer_idx: int
    location: CacheLocation
    is_input: bool  # True if input activation, False if output

    @property
    def memory_bytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


class ActivationCache:
    """
    Cache for storing layer activations during sensitivity analysis.

    By default, stores activations on GPU for fast sequential access.
    Automatically spills to CPU when GPU memory is constrained.

    Usage:
        cache = ActivationCache(device="cuda")

        # During baseline forward pass
        cache.store(layer_idx=0, activation=output_tensor, is_input=False)

        # During sensitivity analysis
        cached_input = cache.get(layer_idx=0, is_input=True)
    """

    def __init__(
        self,
        device: str = "cuda",
        gpu_memory_threshold: float = 0.8,
        enable_cpu_spillover: bool = True,
    ):
        """
        Initialize the activation cache.

        Args:
            device: Primary device for caching (default: "cuda")
            gpu_memory_threshold: Fraction of GPU memory to use before spilling to CPU (0.0-1.0)
            enable_cpu_spillover: Allow spilling activations to CPU when GPU is full
        """
        self.device = device
        self.gpu_memory_threshold = gpu_memory_threshold
        self.enable_cpu_spillover = enable_cpu_spillover

        # Cache storage: key = (layer_idx, is_input), value = CachedActivation
        self._cache: dict[tuple[int, bool], CachedActivation] = {}

        # Statistics
        self.stats = CacheStats()

    def _get_gpu_memory_info(self) -> tuple[int, int, int]:
        """Get GPU memory info: (total, used, free) in bytes."""
        if not torch.cuda.is_available():
            return (0, 0, 0)

        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        free = total - allocated
        return (total, allocated, free)

    def _should_use_gpu(self, tensor_size_bytes: int) -> bool:
        """Check if we should store on GPU based on available memory."""
        if "cuda" not in self.device:
            return False

        total, used, _ = self._get_gpu_memory_info()
        threshold_bytes = total * self.gpu_memory_threshold

        # Use GPU if we're below threshold and have room for this tensor
        return (used + tensor_size_bytes) < threshold_bytes

    def store(
        self,
        layer_idx: int,
        activation: torch.Tensor,
        is_input: bool = False,
        force_location: CacheLocation | None = None,
    ) -> CacheLocation:
        """
        Store an activation tensor in the cache.

        Args:
            layer_idx: Index of the layer
            activation: The activation tensor to cache
            is_input: True if this is layer input, False if output
            force_location: Force storage location (overrides automatic selection)

        Returns:
            The location where the tensor was stored
        """
        key = (layer_idx, is_input)

        # Determine storage location
        tensor_size = activation.numel() * activation.element_size()

        if force_location:
            location = force_location
        elif self._should_use_gpu(tensor_size):
            location = CacheLocation.GPU
        elif self.enable_cpu_spillover:
            location = CacheLocation.CPU
        else:
            # No spillover allowed, try GPU anyway
            location = CacheLocation.GPU

        # Move tensor to appropriate device
        if location == CacheLocation.GPU:
            stored_tensor = activation.detach().to(self.device).clone()
            self.stats.gpu_memory_bytes += tensor_size
        else:
            stored_tensor = activation.detach().cpu().clone()
            self.stats.cpu_memory_bytes += tensor_size

        # Remove old entry if exists
        if key in self._cache:
            old = self._cache[key]
            if old.location == CacheLocation.GPU:
                self.stats.gpu_memory_bytes -= old.memory_bytes
            else:
                self.stats.cpu_memory_bytes -= old.memory_bytes

        # Store
        self._cache[key] = CachedActivation(
            tensor=stored_tensor,
            layer_idx=layer_idx,
            location=location,
            is_input=is_input,
        )
        self.stats.num_entries = len(self._cache)

        return location

    def get(
        self,
        layer_idx: int,
        is_input: bool = False,
        target_device: str | None = None,
    ) -> torch.Tensor | None:
        """
        Retrieve an activation tensor from the cache.

        Args:
            layer_idx: Index of the layer
            is_input: True to get layer input, False for output
            target_device: Device to move tensor to (None = keep current location)

        Returns:
            The cached activation tensor, or None if not found
        """
        key = (layer_idx, is_input)

        if key not in self._cache:
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        cached = self._cache[key]

        if target_device:
            return cached.tensor.to(target_device)
        return cached.tensor

    def has(self, layer_idx: int, is_input: bool = False) -> bool:
        """Check if an activation is cached."""
        return (layer_idx, is_input) in self._cache

    def remove(self, layer_idx: int, is_input: bool = False) -> bool:
        """
        Remove an activation from the cache.

        Returns:
            True if the entry was removed, False if it didn't exist
        """
        key = (layer_idx, is_input)

        if key not in self._cache:
            return False

        cached = self._cache.pop(key)

        # Update stats
        if cached.location == CacheLocation.GPU:
            self.stats.gpu_memory_bytes -= cached.memory_bytes
        else:
            self.stats.cpu_memory_bytes -= cached.memory_bytes

        self.stats.num_entries = len(self._cache)
        return True

    def clear_layer(self, layer_idx: int) -> int:
        """
        Clear all cached activations for a specific layer.

        Returns:
            Number of entries removed
        """
        keys_to_remove = [
            k for k in self._cache.keys()
            if k[0] == layer_idx
        ]

        for key in keys_to_remove:
            self.remove(key[0], key[1])

        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached activations."""
        self._cache.clear()
        self.stats = CacheStats()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def move_to_gpu(self, layer_idx: int | None = None) -> int:
        """
        Move cached activations to GPU.

        Args:
            layer_idx: Specific layer to move, or None for all

        Returns:
            Number of tensors moved
        """
        moved = 0

        for key, cached in self._cache.items():
            if layer_idx is not None and key[0] != layer_idx:
                continue

            if cached.location == CacheLocation.CPU:
                cached.tensor = cached.tensor.to(self.device)
                cached.location = CacheLocation.GPU
                self.stats.cpu_memory_bytes -= cached.memory_bytes
                self.stats.gpu_memory_bytes += cached.memory_bytes
                moved += 1

        return moved

    def move_to_cpu(self, layer_idx: int | None = None) -> int:
        """
        Move cached activations to CPU to free GPU memory.

        Args:
            layer_idx: Specific layer to move, or None for all

        Returns:
            Number of tensors moved
        """
        moved = 0

        for key, cached in self._cache.items():
            if layer_idx is not None and key[0] != layer_idx:
                continue

            if cached.location == CacheLocation.GPU:
                cached.tensor = cached.tensor.cpu()
                cached.location = CacheLocation.CPU
                self.stats.gpu_memory_bytes -= cached.memory_bytes
                self.stats.cpu_memory_bytes += cached.memory_bytes
                moved += 1

        return moved

    def get_memory_summary(self) -> str:
        """Get a human-readable memory summary."""
        _, _, free = self._get_gpu_memory_info()

        return (
            f"ActivationCache: {self.stats.num_entries} entries | "
            f"GPU: {self.stats.gpu_memory_gb:.2f} GB | "
            f"CPU: {self.stats.cpu_memory_gb:.2f} GB | "
            f"GPU Available: {free / (1024**3):.2f} GB"
        )

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: tuple[int, bool]) -> bool:
        return key in self._cache

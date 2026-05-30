"""Distributed training utilities for QAT.

Thin wrappers around torchrun env vars (LOCAL_RANK, WORLD_SIZE, RANK).
Safe to import in single-GPU mode — returns sensible defaults.
"""

from __future__ import annotations

import os


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return world_size() > 1


def is_rank0() -> bool:
    return rank() == 0

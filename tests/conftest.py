"""
Pytest configuration and fixtures for Quanto tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def small_model_path():
    """Path to a small model for testing. Override with --model-path option."""
    return None


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory for test outputs."""
    return tmp_path / "output"


@pytest.fixture
def device():
    """Device to use for testing."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"

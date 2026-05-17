"""
Evaluation utilities for QAT trials.

Wraps Quark's ppl_eval for perplexity computation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.contrib.llm_eval import ppl_eval


def compute_perplexity(
    model,
    tokenizer,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity on wikitext-2 test set.

    Args:
        model: The model to evaluate (with fake or real quantization applied).
        tokenizer: HuggingFace tokenizer.
        device: Device string.

    Returns:
        Perplexity score (float).
    """
    model.eval()
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    with torch.no_grad():
        ppl = ppl_eval(model, testenc, device)

    return float(ppl)

"""
Evaluation utilities for QAT trials.

Computes perplexity using sliding-window approach.
Direct PyTorch implementation — avoids broken lm_eval compatibility in quark.contrib.
"""

from __future__ import annotations

import torch
from datasets import load_dataset


def compute_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_subset: str | None = "wikitext-2-raw-v1",
    dataset_split: str = "test",
    stride: int = 512,
    text_column: str = "text",
) -> float:
    """
    Compute perplexity using sliding window.

    Args:
        model: Model to evaluate.
        tokenizer: HuggingFace tokenizer.
        dataset_name: HuggingFace dataset name.
        dataset_subset: Dataset subset (e.g. "wikitext-2-raw-v1").
        dataset_split: Dataset split (e.g. "test", "validation").
        stride: Sliding window stride.

    Returns:
        Perplexity score (float).
    """
    model.eval()
    if hasattr(model, "hf_device_map"):
        first_device = next(iter(model.hf_device_map.values()))
        device = torch.device(first_device)
    else:
        device = next(model.parameters()).device

    testdata = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    test_text = "\n\n".join(item for item in testdata[text_column] if item)
    testenc = tokenizer(test_text, return_tensors="pt")
    test_ids = testenc.input_ids.to(device)

    seq_len = test_ids.shape[1]
    nlls = []
    prev_end = 0

    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + stride, seq_len)
            trg_len = end - prev_end
            input_chunk = test_ids[:, begin:end]
            target_chunk = input_chunk.clone()
            target_chunk[:, :-trg_len] = -100

            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)
            prev_end = end

    if seq_len == 0:
        return float("inf")
    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return float(ppl)

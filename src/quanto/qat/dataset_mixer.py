"""
Multi-dataset mixing for QAT training.

Loads multiple HuggingFace datasets, tokenizes them,
and combines them according to specified ratios.
"""

from __future__ import annotations

import logging

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Known dataset aliases and their HuggingFace identifiers
_DATASET_ALIASES = {
    "wikitext": ("wikitext", "wikitext-2-raw-v1"),
    "pileval": ("mit-han-lab/pile-val-backup", None),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None),
    "cnn_dailymail": ("cnn_dailymail", "3.0.0"),
}


class TokenBlockDataset(Dataset):
    """Tokenized text split into fixed-length blocks."""

    def __init__(
        self,
        token_ids: list[list[int]],
        seq_len: int,
    ):
        self.blocks = []
        current_block: list[int] = []
        for ids in token_ids:
            current_block.extend(ids)
            while len(current_block) >= seq_len:
                block = current_block[:seq_len]
                self.blocks.append(
                    {
                        "input_ids": torch.tensor(block, dtype=torch.long),
                        "labels": torch.tensor(block, dtype=torch.long),
                        "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    }
                )
                current_block = current_block[seq_len:]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.blocks[idx]


class MixedDataset(Dataset):
    """
    A dataset that mixes multiple HuggingFace datasets according to ratios.

    Each dataset is tokenized and split into fixed-length blocks,
    then combined proportionally.
    """

    def __init__(
        self,
        datasets: list[dict],
        tokenizer,
        seq_len: int = 512,
        ratios: list[float] | None = None,
        total_samples: int | None = None,
        seed: int = 42,
    ):
        """
        Args:
            datasets: List of dicts with 'name' key (and optional 'split', 'text_column').
            tokenizer: HuggingFace tokenizer.
            seq_len: Block length for tokenized sequences.
            ratios: Mixing ratios (must sum to 1.0). If None, use equal ratios.
            total_samples: Target total sample count. If None, use all available.
            seed: Random seed for shuffling.
        """
        if ratios is None:
            ratios = [1.0 / len(datasets)] * len(datasets)

        # Normalize ratios
        ratio_sum = sum(ratios)
        ratios = [r / ratio_sum for r in ratios]

        # Load and tokenize each dataset
        sub_datasets = []
        for ds_spec, ratio in zip(datasets, ratios):
            name = ds_spec["name"]
            split = ds_spec.get("split", "train")
            text_column = ds_spec.get("text_column", "text")

            hf_name, subset = _resolve_dataset_name(name)
            logger.info(f"Loading dataset: {name} (ratio={ratio:.2f})")

            ds = load_dataset(hf_name, subset, split=split)

            # Sample if max_samples is set
            max_samples = ds_spec.get("max_samples")
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))

            # Extract text and tokenize
            texts = [item[text_column] for item in ds if item[text_column]]
            token_ids = _tokenize_texts(texts, tokenizer)

            block_ds = TokenBlockDataset(token_ids, seq_len)
            sub_datasets.append((block_ds, ratio))

        # Allocate samples proportionally
        if total_samples is None:
            # Use the minimum proportional to the smallest dataset
            min_blocks = min(len(ds) for ds, _ in sub_datasets)
            total_samples = int(min_blocks / min(r for _, r in sub_datasets))

        self.samples: list[dict[str, torch.Tensor]] = []
        for block_ds, ratio in sub_datasets:
            n = min(int(total_samples * ratio), len(block_ds))
            for i in range(n):
                self.samples.append(block_ds[i])

        # Shuffle
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(self.samples), generator=generator).tolist()
        self.samples = [self.samples[i] for i in indices]

        logger.info(
            f"MixedDataset: {len(self.samples)} total samples from {len(datasets)} datasets"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def _resolve_dataset_name(name: str) -> tuple[str, str | None]:
    """Resolve dataset alias to (hf_name, subset)."""
    if name in _DATASET_ALIASES:
        return _DATASET_ALIASES[name]
    return (name, None)


def _tokenize_texts(texts: list[str], tokenizer) -> list[list[int]]:
    """Tokenize a list of texts into token ID lists."""
    if not texts:
        return []
    encodings = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    return encodings["input_ids"]


def normalize_ratios(raw_ratios: list[float], min_ratio: float = 0.1) -> list[float]:
    """
    Normalize raw ratio values to sum to 1.0 with a minimum per-dataset.

    Uses softmax for normalization, then clips to min_ratio.
    """
    import math

    # Softmax normalization
    max_val = max(raw_ratios)
    exp_vals = [math.exp(v - max_val) for v in raw_ratios]
    exp_sum = sum(exp_vals)
    ratios = [e / exp_sum for e in exp_vals]

    # Enforce minimum ratio
    needs_adjust = any(r < min_ratio for r in ratios)
    if needs_adjust:
        ratios = [max(r, min_ratio) for r in ratios]
        ratio_sum = sum(ratios)
        ratios = [r / ratio_sum for r in ratios]

    return ratios

"""
Calibration Data Manager: Handle calibration dataset loading and preprocessing.

Supports both HuggingFace datasets and local datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from quark.shares.utils.import_utils import is_transformers_available

if is_transformers_available():
    from transformers import PreTrainedTokenizer


class LocalTextDataset(Dataset):
    """Dataset for loading text from local files."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        num_samples: int = 512,
    ):
        """
        Initialize local text dataset.

        Args:
            data_path: Path to text file or directory containing text files
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            num_samples: Number of samples to create
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples

        # Load text data
        data_path = Path(data_path)
        texts = []

        if data_path.is_file():
            with open(data_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        elif data_path.is_dir():
            for file_path in data_path.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.extend([line.strip() for line in f if line.strip()])
            for file_path in data_path.glob("*.json"):
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                # Common keys for text data
                                for key in ["text", "content", "prompt", "input"]:
                                    if key in item:
                                        texts.append(str(item[key]))
                                        break
                            elif isinstance(item, str):
                                texts.append(item)
        else:
            raise ValueError(f"Data path {data_path} does not exist")

        if not texts:
            raise ValueError(f"No text data found in {data_path}")

        print(f"[INFO] Loaded {len(texts)} text segments from {data_path}")

        # Tokenize and create samples
        self.samples = self._create_samples(texts)

    def _create_samples(self, texts: list[str]) -> list[dict[str, torch.Tensor]]:
        """Create samples from texts."""
        samples = []
        current_tokens: list[int] = []

        for text in tqdm(texts, desc="Processing calibration data"):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            current_tokens.extend(tokens)

            # Create samples when we have enough tokens
            while len(current_tokens) >= self.max_length:
                sample_tokens = current_tokens[:self.max_length]
                current_tokens = current_tokens[self.max_length:]

                input_ids = torch.tensor([sample_tokens], dtype=torch.long)
                samples.append({"input_ids": input_ids})

                if len(samples) >= self.num_samples:
                    return samples

        # Handle remaining tokens
        if current_tokens and len(samples) < self.num_samples:
            # Pad to max_length
            if len(current_tokens) < self.max_length:
                current_tokens = current_tokens + [self.tokenizer.pad_token_id or 0] * (self.max_length - len(current_tokens))
            input_ids = torch.tensor([current_tokens[:self.max_length]], dtype=torch.long)
            samples.append({"input_ids": input_ids})

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class CalibrationDataManager:
    """
    Manages calibration data loading from various sources.
    """

    SUPPORTED_DATASETS = [
        "pileval",  # mit-han-lab/pile-val-backup
        "wikitext",  # wikitext-2-raw-v1
        "cnn_dailymail",
        "local",  # Local file/directory
    ]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        seq_len: int = 512,
        num_samples: int = 512,
        batch_size: int = 1,
        device: str = "cuda",
    ):
        """
        Initialize calibration data manager.

        Args:
            tokenizer: Tokenizer for encoding text
            seq_len: Sequence length for calibration
            num_samples: Number of calibration samples
            batch_size: Batch size for dataloader
            device: Device to load data to
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device

    def load_from_huggingface(self, dataset_name: str = "pileval") -> DataLoader:
        """
        Load calibration data from HuggingFace Hub.

        Args:
            dataset_name: Name of the dataset ('pileval', 'wikitext', 'cnn_dailymail')

        Returns:
            DataLoader with calibration data
        """
        from datasets import load_dataset

        print(f"[INFO] Loading calibration data from HuggingFace: {dataset_name}")

        if dataset_name == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            texts = [t for t in dataset["text"] if t and str(t).strip()][:self.num_samples]
        elif dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in dataset["text"] if t and str(t).strip()][:self.num_samples]
        elif dataset_name == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
            texts = [t for t in dataset["article"] if t and str(t).strip()][:self.num_samples]
        else:
            # Try to load any dataset from HuggingFace
            try:
                dataset = load_dataset(dataset_name, split="train")
                # Find text column
                text_col = None
                if hasattr(dataset, 'column_names'):
                    for col in ["text", "content", "prompt", "input", "article"]:
                        if col in dataset.column_names:
                            text_col = col
                            break
                    if text_col is None and dataset.column_names:
                        text_col = dataset.column_names[0]
                if text_col:
                    texts = [str(t) for t in dataset[text_col] if t and str(t).strip()][:self.num_samples]
                else:
                    raise ValueError(f"Could not find text column in dataset")
            except Exception as e:
                raise ValueError(f"Could not load dataset '{dataset_name}': {e}")

        # Tokenize all texts at once (like quark does)
        batch_encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.seq_len
        )

        # Move to device
        if self.device:
            batch_encoded = batch_encoded.to(self.device)

        # Get input_ids tensor
        input_ids = batch_encoded["input_ids"]

        # Create DataLoader
        calib_dataloader = DataLoader(input_ids, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return calib_dataloader

    def load_from_local(self, data_path: str | Path) -> DataLoader:
        """
        Load calibration data from local files.

        Args:
            data_path: Path to text file or directory

        Returns:
            DataLoader with calibration data
        """
        print(f"[INFO] Loading calibration data from local path: {data_path}")

        dataset = LocalTextDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.seq_len,
            num_samples=self.num_samples,
        )

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def load_from_path_or_name(self, data_source: str) -> DataLoader:
        """
        Load calibration data from either local path or HuggingFace dataset name.

        Args:
            data_source: Either a local path or HuggingFace dataset name

        Returns:
            DataLoader with calibration data
        """
        path = Path(data_source)
        if path.exists():
            return self.load_from_local(data_source)
        elif data_source in self.SUPPORTED_DATASETS:
            return self.load_from_huggingface(data_source)
        else:
            # Try to load from HuggingFace directly
            try:
                return self.load_from_huggingface(data_source)
            except Exception as e:
                raise ValueError(
                    f"Could not load calibration data from '{data_source}'. "
                    f"Either provide a valid local path or HuggingFace dataset name. "
                    f"Error: {e}"
                )

    def _create_samples_from_texts(self, texts: list[str]) -> list[torch.Tensor]:
        """Create samples from a list of texts."""
        samples = []
        current_tokens: list[int] = []

        for text in tqdm(texts, desc="Processing texts"):
            if not text or not text.strip():
                continue

            tokens = self.tokenizer.encode(text.strip(), add_special_tokens=False)
            current_tokens.extend(tokens)

            while len(current_tokens) >= self.seq_len:
                sample_tokens = current_tokens[:self.seq_len]
                current_tokens = current_tokens[self.seq_len:]

                input_ids = torch.tensor(sample_tokens, dtype=torch.long)
                samples.append(input_ids)

                if len(samples) >= self.num_samples:
                    return samples

        return samples[:self.num_samples]


def get_calib_dataloader(
    dataset_name_or_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    num_calib_data: int = 512,
    seqlen: int = 512,
    device: str = "cuda",
) -> DataLoader:
    """
    Get calibration dataloader from dataset name or local path.

    This is a convenience function that wraps CalibrationDataManager.

    Args:
        dataset_name_or_path: HuggingFace dataset name or local path
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        num_calib_data: Number of calibration samples
        seqlen: Sequence length
        device: Device

    Returns:
        DataLoader for calibration
    """
    manager = CalibrationDataManager(
        tokenizer=tokenizer,
        seq_len=seqlen,
        num_samples=num_calib_data,
        batch_size=batch_size,
        device=device,
    )

    return manager.load_from_path_or_name(dataset_name_or_path)

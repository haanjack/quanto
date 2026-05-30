"""
QAT Trial Runner — HFQATTrainer adapter.

Implements the QATTrainer protocol for HuggingFace Transformers.
Orchestrates the full PTQ -> fake-quant -> QAT -> evaluate -> export pipeline.
"""

from __future__ import annotations

import gc
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import QATSearchConfig
from .dataset_mixer import MixedDataset, normalize_ratios
from .distributed import is_rank0, local_rank, world_size
from .evaluate import compute_perplexity
from .quantize import apply_fake_quant
from .train import train_qat
from .trainer_interface import MetricCallback, TrainResult

logger = logging.getLogger(__name__)


class HFQATTrainer:
    """HuggingFace implementation of the QATTrainer protocol."""

    def __init__(self, search_config: QATSearchConfig, trial_id: str | None = None):
        self.search_config = search_config
        self.trial_id = trial_id or "trial-0"
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.sampled_config = {}
        self.train_dataset = None
        self.eval_dataset = None
        self._epochs_trained = 0

    def initialize(self, sampled_config: dict, is_resume: bool = False) -> None:
        """Load model, apply PTQ, build datasets."""
        self.sampled_config = sampled_config
        cfg = self.search_config

        logger.info(f"[{self.trial_id}] Loading model from {cfg.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.cuda.set_device(local_rank())
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=cfg.trust_remote_code,
        ).to(f"cuda:{local_rank()}")

        # Apply fake quantization (PTQ)
        precision = sampled_config.get("precision", "int4")
        group_size = sampled_config.get("group_size", 128)
        symmetric = sampled_config.get("symmetric", True)

        logger.info(
            f"[{self.trial_id}] Applying fake quantization: "
            f"precision={precision}, group_size={group_size}, symmetric={symmetric}"
        )
        print(
            f"[{self.trial_id}] Before PTQ: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated",
            flush=True,
        )

        self.model = apply_fake_quant(
            model=self.model,
            tokenizer=self.tokenizer,
            precision=precision,
            group_size=group_size,
            symmetric=symmetric,
            calibration_dataset=sampled_config.get("calibration_dataset", "wikitext"),
            num_calib_samples=1 if is_resume else sampled_config.get("num_calib_samples", 128),
            seq_len=cfg.seq_len,
            device=f"cuda:{local_rank()}",
        )

        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"[{self.trial_id}] After PTQ+GC: "
            f"{torch.cuda.memory_allocated() / 1e9:.2f}GB allocated",
            flush=True,
        )

        # Build datasets
        self.train_dataset = self._build_train_datasets()
        self.eval_dataset = MixedDataset(
            datasets=[
                {
                    "name": cfg.eval_dataset.name,
                    "split": cfg.eval_dataset.split,
                    "subset": cfg.eval_dataset.subset,
                }
            ],
            tokenizer=self.tokenizer,
            seq_len=cfg.seq_len,
            total_samples=cfg.max_eval_samples,
        )

    def train_segment(
        self,
        hyperparams: dict,
        num_epochs: int,
        metric_callback: MetricCallback,
        resume_from: str | None = None,
    ) -> TrainResult:
        """Train for num_epochs, reporting metrics via callback."""
        if self.model is None:
            raise RuntimeError("Must call initialize() before train_segment()")

        # Load checkpoint if resuming (restores scale params)
        if resume_from is not None:
            self._load_scales(resume_from)

        output_dir = resume_from or os.path.join(
            self.search_config.output_dir, "trials", self.trial_id
        )

        # Compute gradient_accumulation_steps from global_batch_size
        per_device_bs = hyperparams.get("per_device_train_batch_size", 1)
        ws = world_size()
        if "global_batch_size" in hyperparams:
            gbs = hyperparams["global_batch_size"]
            denom = per_device_bs * ws
            if gbs < denom:
                logger.warning(
                    f"global_batch_size={gbs} < per_device_bs*world_size={denom}, "
                    f"clamping gradient_accumulation_steps to 1"
                )
                grad_accum = 1
            elif gbs % denom != 0:
                grad_accum = gbs // denom
                logger.warning(
                    f"global_batch_size={gbs} not divisible by "
                    f"per_device_bs*world_size={denom}, "
                    f"rounding down to gradient_accumulation_steps={grad_accum}"
                )
            else:
                grad_accum = gbs // denom
        else:
            grad_accum = hyperparams.get("gradient_accumulation_steps", 4)

        self.trainer = train_qat(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            learning_rate=hyperparams.get("learning_rate", 2e-5),
            num_epochs=num_epochs,
            per_device_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            weight_decay=hyperparams.get("weight_decay", 0.0),
            warmup_ratio=hyperparams.get("warmup_ratio", 0.0),
            gradient_checkpointing=True,
            only_train_scaling_factor=hyperparams.get("only_train_scaling_factor", False),
            precision=hyperparams.get("precision", "int4"),
            output_dir=output_dir,
            metric_callback=metric_callback,
        )

        self._epochs_trained += num_epochs

        max_epochs = hyperparams.get("num_train_epochs", 999)
        return TrainResult(
            metrics=metric_callback.last(),
            epoch=self._epochs_trained,
            finished=self._epochs_trained >= max_epochs,
        )

    def evaluate(self) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("No model loaded")
        cfg = self.search_config
        ppl = compute_perplexity(
            self.model,
            self.tokenizer,
            dataset_name=cfg.eval_dataset.name,
            dataset_subset=cfg.eval_dataset.subset,
            dataset_split=cfg.eval_dataset.split,
            text_column=cfg.eval_dataset.text_column,
        )
        return {"perplexity": ppl}

    def save_checkpoint(self, path: str) -> None:
        if not is_rank0():
            return
        os.makedirs(path, exist_ok=True)

        # Save trainable scales
        scale_state = {
            name: param.data.cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        torch.save(scale_state, os.path.join(path, "scales.pt"))

        # Save optimizer state
        if self.trainer and self.trainer.optimizer:
            torch.save(
                self.trainer.optimizer.state_dict(),
                os.path.join(path, "optimizer.pt"),
            )

        logger.info(
            f"[{self.trial_id}] Saved checkpoint: {len(scale_state)} trainable params to {path}"
        )

    def load_checkpoint(self, path: str) -> None:
        self._load_scales(path)

    def _load_scales(self, path: str) -> None:
        """Load scale parameters from checkpoint."""
        scale_path = os.path.join(path, "scales.pt")
        if not os.path.exists(scale_path):
            logger.warning(f"No scales.pt in {path}")
            return
        scale_state = torch.load(scale_path, weights_only=True)
        for name, tensor in scale_state.items():
            parts = name.split(".")
            mod = self.model
            for part in parts[:-1]:
                mod = getattr(mod, part)
            param = getattr(mod, parts[-1])
            param.data.copy_(tensor.to(param.device))
        logger.info(f"[{self.trial_id}] Loaded scales from {path}")

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
        if self.trainer is not None:
            del self.trainer
        self.model = None
        self.trainer = None
        torch.cuda.empty_cache()
        gc.collect()

    def _build_train_datasets(self) -> MixedDataset:
        """Build the mixed training dataset from config."""
        cfg = self.search_config
        ds_ratio_search = cfg.dataset_ratio_search

        if ds_ratio_search is not None:
            raw_ratios = []
            for i in range(len(ds_ratio_search.datasets)):
                raw_ratios.append(
                    self.sampled_config.get(
                        f"ds_ratio_{i}",
                        1.0 / len(ds_ratio_search.datasets),
                    )
                )
            ratios = normalize_ratios(raw_ratios, ds_ratio_search.min_ratio)
            datasets = [{"name": name} for name in ds_ratio_search.datasets]
            return MixedDataset(
                datasets=datasets,
                tokenizer=self.tokenizer,
                seq_len=cfg.seq_len,
                ratios=ratios,
                total_samples=cfg.max_train_samples,
            )

        # Fixed ratios from config
        datasets = [
            {"name": ds.name, "ratio": ds.ratio, "subset": ds.subset} for ds in cfg.train_datasets
        ]
        ratios = [ds["ratio"] for ds in datasets]
        return MixedDataset(
            datasets=datasets,
            tokenizer=self.tokenizer,
            seq_len=cfg.seq_len,
            ratios=ratios,
            total_samples=cfg.max_train_samples,
        )

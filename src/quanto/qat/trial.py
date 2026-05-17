"""
QAT Trial Runner.

The function Ray Tune calls per trial. Orchestrates the full
PTQ → fake-quant → QAT → evaluate → export pipeline.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import QATSearchConfig
from .dataset_mixer import MixedDataset, normalize_ratios
from .evaluate import compute_perplexity
from .export import export_quantized_model
from .quantize import apply_fake_quant
from .train import train_qat

logger = logging.getLogger(__name__)


def qat_trial(sampled_config: dict, search_config: QATSearchConfig):
    """
    Ray Tune trainable. Called once per trial.

    Args:
        sampled_config: Ray-sampled hyperparameters for this trial.
        search_config: Global search configuration.
    """
    import ray.tune

    trial_id = ray.tune.get_trial_id()
    trial_dir = os.path.join(search_config.output_dir, "trials", trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    timing = {}
    t_start = time.time()

    try:
        # Step 1: Load model and tokenizer
        logger.info(f"[{trial_id}] Loading model from {search_config.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            search_config.model_path,
            trust_remote_code=search_config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            search_config.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=search_config.trust_remote_code,
            device_map="auto",
        )

        # Step 2: Apply fake quantization (PTQ)
        precision = sampled_config.get("precision", "int4")
        group_size = sampled_config.get("group_size", 128)
        symmetric = sampled_config.get("symmetric", True)

        logger.info(f"[{trial_id}] Applying fake quantization: precision={precision}, group_size={group_size}, symmetric={symmetric}")
        t_ptq = time.time()
        model = apply_fake_quant(
            model=model,
            tokenizer=tokenizer,
            precision=precision,
            group_size=group_size,
            symmetric=symmetric,
            calibration_dataset=sampled_config.get("calibration_dataset", "wikitext"),
            num_calib_samples=sampled_config.get("num_calib_samples", 128),
            seq_len=search_config.seq_len,
            device="cuda",
        )
        timing["ptq"] = time.time() - t_ptq

        # Step 3: Build mixed training dataset
        train_datasets = _build_train_datasets(search_config, sampled_config, tokenizer)
        eval_dataset = MixedDataset(
            datasets=[{"name": search_config.eval_dataset.name, "split": search_config.eval_dataset.split}],
            tokenizer=tokenizer,
            seq_len=search_config.seq_len,
            total_samples=search_config.max_eval_samples,
        )

        # Step 4: QAT fine-tuning
        logger.info(f"[{trial_id}] Starting QAT training")
        t_train = time.time()
        trainer = train_qat(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_datasets,
            eval_dataset=eval_dataset,
            learning_rate=sampled_config.get("learning_rate", 2e-5),
            num_epochs=sampled_config.get("num_train_epochs", 3),
            per_device_batch_size=sampled_config.get("per_device_train_batch_size", 2),
            weight_decay=sampled_config.get("weight_decay", 0.0),
            warmup_ratio=sampled_config.get("warmup_ratio", 0.0),
            only_train_scaling_factor=sampled_config.get("only_train_scaling_factor", False),
            precision=precision,
            output_dir=trial_dir,
        )
        timing["train"] = time.time() - t_train

        # Step 5: Evaluate
        logger.info(f"[{trial_id}] Evaluating")
        t_eval = time.time()
        model = trainer.model
        ppl = compute_perplexity(model, tokenizer, device="cuda")
        timing["eval"] = time.time() - t_eval

        eval_loss = trainer.state.best_metric if trainer.state.best_metric is not None else float("inf")

        logger.info(f"[{trial_id}] perplexity={ppl:.4f}, eval_loss={eval_loss:.4f}")

        # Step 6: Export if target met
        exported = False
        threshold = search_config.target.threshold
        if threshold is not None and ppl <= threshold:
            best_dir = os.path.join(search_config.output_dir, "best_model")
            logger.info(f"[{trial_id}] Target met! Exporting to {best_dir}")
            export_quantized_model(
                model=model,
                tokenizer=tokenizer,
                output_dir=best_dir,
                weight_format=search_config.export_weight_format,
            )
            exported = True

        # Step 7: Save trial result
        timing["total"] = time.time() - t_start
        trial_result = {
            "trial_id": trial_id,
            "sampled_config": sampled_config,
            "perplexity": ppl,
            "eval_loss": eval_loss,
            "timing": timing,
            "exported": exported,
        }
        result_path = os.path.join(trial_dir, "trial_result.json")
        with open(result_path, "w") as f:
            json.dump(trial_result, f, indent=2, default=str)

        # Report to Ray Tune
        ray.tune.report(perplexity=ppl, eval_loss=eval_loss)

    except Exception as e:
        logger.error(f"[{trial_id}] Trial failed: {e}")
        ray.tune.report(perplexity=float("inf"), eval_loss=float("inf"), error=str(e))

    finally:
        # Cleanup GPU memory
        for var_name in ("model", "trainer"):
            if var_name in locals():
                del locals()[var_name]
        torch.cuda.empty_cache()
        gc.collect()


def _build_train_datasets(
    search_config: QATSearchConfig,
    sampled_config: dict,
    tokenizer,
) -> MixedDataset:
    """Build the mixed training dataset from config, applying searched ratios if applicable."""
    ds_ratio_search = search_config.dataset_ratio_search

    if ds_ratio_search is not None:
        # Ratios are being searched — extract from sampled_config
        raw_ratios = []
        for i, ds_name in enumerate(ds_ratio_search.datasets):
            raw_ratios.append(sampled_config.get(f"ds_ratio_{i}", 1.0 / len(ds_ratio_search.datasets)))
        ratios = normalize_ratios(raw_ratios, ds_ratio_search.min_ratio)

        datasets = [{"name": name} for name in ds_ratio_search.datasets]
        return MixedDataset(
            datasets=datasets,
            tokenizer=tokenizer,
            seq_len=search_config.seq_len,
            ratios=ratios,
            total_samples=search_config.max_train_samples,
        )

    # Fixed ratios from config
    datasets = [{"name": ds.name, "ratio": ds.ratio} for ds in search_config.train_datasets]
    ratios = [ds["ratio"] for ds in datasets]

    return MixedDataset(
        datasets=datasets,
        tokenizer=tokenizer,
        seq_len=search_config.seq_len,
        ratios=ratios,
        total_samples=search_config.max_train_samples,
    )

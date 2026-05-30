"""
Trainer-agnostic PBT (Population-Based Training) hyperparameter tuner.

Replaces the Ray Tune integration. Runs population members sequentially
on a single GPU (or in lockstep across DDP ranks), with exploit/explore
at configurable intervals.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any

from .config import QATSearchConfig
from .distributed import is_distributed, is_rank0, local_rank
from .population import (
    PopulationMember,
    clone_checkpoint,
    load_population_state,
    save_population_state,
)
from .sampler import sample_initial_population
from .trainer_interface import (
    MetricCallback,
    build_sinks,
    close_sinks,
)

logger = logging.getLogger(__name__)


def _perturb_hyperparams(
    hyperparams: dict[str, Any],
    search_space: dict,
    perturbation_factor: float,
) -> dict[str, Any]:
    """Perturb continuous hyperparameters by multiplying by (1 ± perturbation_factor).

    Categorical hyperparameters are left unchanged.
    """
    perturbed = copy.deepcopy(hyperparams)
    for name, value in perturbed.items():
        dim = search_space.get(name)
        if dim is None:
            continue
        if dim.choices is not None:
            continue
        if not isinstance(value, (int, float)):
            continue
        factor = 1.0 + random.uniform(-perturbation_factor, perturbation_factor)
        perturbed_val = value * factor
        if isinstance(value, int):
            perturbed_val = round(perturbed_val)
        perturbed[name] = perturbed_val
        if dim.min is not None:
            perturbed[name] = max(perturbed[name], dim.min)
        if dim.max is not None:
            perturbed[name] = min(perturbed[name], dim.max)
    return perturbed


def _exploit_explore(
    population: list[PopulationMember],
    search_space: dict,
    mode: str,
    perturbation_factor: float,
) -> None:
    """PBT exploit/explore step.

    Bottom 1/3 clones top 1/3's checkpoint and gets perturbed hyperparams.
    """
    sorted_members = sorted(
        population,
        key=lambda m: m.best_metric,
        reverse=(mode == "max"),
    )

    n = len(sorted_members)
    top_k = max(1, n // 3)
    bottom_k = max(1, n // 3)

    top_performers = sorted_members[:top_k]
    bottom_performers = sorted_members[-bottom_k:]

    for underperformer in bottom_performers:
        donor = random.choice(top_performers)
        if underperformer.member_id == donor.member_id:
            continue

        logger.info(
            f"Exploit: member {underperformer.member_id} cloning from "
            f"member {donor.member_id} (best_metric={donor.best_metric:.4f})"
        )

        # Copy checkpoint files (rank 0 only)
        if is_rank0():
            clone_checkpoint(donor, underperformer)

        # Explore: perturb hyperparams from donor
        underperformer.hyperparams = _perturb_hyperparams(
            donor.hyperparams, search_space, perturbation_factor
        )

        # Save updated hyperparams (rank 0 only)
        if is_rank0():
            config_path = os.path.join(underperformer.checkpoint_path, "hyperparams.json")
            with open(config_path, "w") as f:
                json.dump(underperformer.hyperparams, f, indent=2, default=str)

        underperformer.finished = False


def _should_stop(
    population: list[PopulationMember],
    config: QATSearchConfig,
    round_num: int,
) -> bool:
    """Check stopping criteria."""
    target = config.target

    # All members finished
    if all(m.finished for m in population):
        logger.info("All population members finished training")
        return True

    # Target threshold met
    if target.threshold is not None:
        for m in population:
            if target.mode == "min" and m.best_metric <= target.threshold:
                logger.info(
                    f"Target met: member {m.member_id} "
                    f"{target.metric}={m.best_metric:.4f} <= {target.threshold}"
                )
                return True
            if target.mode == "max" and m.best_metric >= target.threshold:
                logger.info(
                    f"Target met: member {m.member_id} "
                    f"{target.metric}={m.best_metric:.4f} >= {target.threshold}"
                )
                return True

    # Max rounds
    if round_num >= target.max_trials:
        logger.info(f"Max rounds reached: {round_num} >= {target.max_trials}")
        return True

    return False


def run_pbt(
    config: QATSearchConfig,
    trainer_factory: Callable[[QATSearchConfig], Any],
    resume: bool = False,
) -> dict:
    """Main PBT loop.

    For each exploit interval:
      1. For each population member (sequentially):
         a. Create trainer, load checkpoint if resuming
         b. Train for exploit_interval epochs
         c. Evaluate, record metric
         d. Save checkpoint
         e. Cleanup trainer (free GPU memory)
      2. Exploit/explore step
      3. Save state for resume

    Returns search summary dict.
    """
    tuner_cfg = config.tuner_config
    target = config.target
    exploit_interval = tuner_cfg.exploit_interval

    os.makedirs(config.output_dir, exist_ok=True)
    if is_distributed():
        import torch
        import torch.distributed

        torch.cuda.set_device(local_rank())
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier()
    state_path = os.path.join(config.output_dir, "pbt_state.json")

    # Synchronize random seed so all DDP ranks produce identical
    # hyperparams and perturbations
    random.seed(42)

    # Load or create population
    if resume and os.path.exists(state_path):
        population, round_num = load_population_state(state_path)
        logger.info(f"Resuming PBT from round {round_num} with {len(population)} members")
    else:
        population = sample_initial_population(
            search_space=config.search_space,
            population_size=tuner_cfg.population_size,
            output_dir=config.output_dir,
            dataset_ratio_search=config.dataset_ratio_search,
        )
        round_num = 0

    best_overall = float("inf") if target.mode == "min" else float("-inf")
    best_config: dict | None = None
    t_start = time.time()

    try:
        while True:
            if _should_stop(population, config, round_num):
                break

            # Time limit check
            if target.max_total_time_seconds:
                elapsed = time.time() - t_start
                if elapsed >= target.max_total_time_seconds:
                    logger.info(f"Time limit reached: {elapsed:.0f}s")
                    break

            round_num += 1
            logger.info(f"=== PBT Round {round_num} ===")

            # Phase 1: Train each member for exploit_interval epochs
            for member in population:
                if member.finished:
                    continue

                logger.info(
                    f"Training member {member.member_id} "
                    f"(epochs {member.total_epochs_trained} -> "
                    f"{member.total_epochs_trained + exploit_interval})"
                )

                # Create per-member metric sinks (rank 0 only to avoid file conflicts)
                sinks = (
                    build_sinks(
                        member_id=member.member_id,
                        tracking_config=tuner_cfg.tracking,
                        output_dir=config.output_dir,
                        hyperparams=member.hyperparams,
                    )
                    if is_rank0()
                    else []
                )
                metric_callback = MetricCallback(sinks=sinks)

                trainer = trainer_factory(config)

                try:
                    trainer.initialize(
                        member.hyperparams, is_resume=member.total_epochs_trained > 0
                    )

                    checkpoint = None
                    if member.total_epochs_trained > 0:
                        checkpoint = member.checkpoint_path
                        trainer.load_checkpoint(checkpoint)

                    trainer.train_segment(
                        hyperparams=member.hyperparams,
                        num_epochs=exploit_interval,
                        metric_callback=metric_callback,
                        resume_from=checkpoint,
                    )

                    # Get metric
                    metric_name = target.metric
                    if metric_name in metric_callback.last():
                        metric_val = metric_callback.last()[metric_name]
                    else:
                        eval_metrics = trainer.evaluate()
                        metric_val = eval_metrics.get(metric_name, float("inf"))

                    member.record_metric(metric_val, target.mode)
                    member.total_epochs_trained += exploit_interval

                    # Check if training reached max epochs
                    max_epochs = member.hyperparams.get("num_train_epochs", 999)
                    if member.total_epochs_trained >= max_epochs:
                        member.finished = True

                    # Early stopping patience
                    if (
                        tuner_cfg.early_stopping_patience > 0
                        and len(member.metric_history) >= tuner_cfg.early_stopping_patience
                    ):
                        recent = member.metric_history[-tuner_cfg.early_stopping_patience :]
                        if target.mode == "min" and all(r >= member.best_metric for r in recent):
                            logger.info(
                                f"Early stopping member {member.member_id}: "
                                f"no improvement for {tuner_cfg.early_stopping_patience} evals"
                            )
                            member.finished = True

                    # Save checkpoint
                    trainer.save_checkpoint(member.checkpoint_path)

                    # Track overall best
                    is_better = (target.mode == "min" and metric_val < best_overall) or (
                        target.mode == "max" and metric_val > best_overall
                    )
                    if is_better:
                        best_overall = metric_val
                        best_config = copy.deepcopy(member.hyperparams)

                    logger.info(
                        f"Member {member.member_id}: "
                        f"{metric_name}={metric_val:.4f} "
                        f"(best={member.best_metric:.4f})"
                    )

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(f"Member {member.member_id} failed: {e}")
                    fallback = float("inf") if target.mode == "min" else float("-inf")
                    member.record_metric(fallback, target.mode)
                finally:
                    trainer.cleanup()
                    close_sinks(sinks)

            # Phase 2: Exploit/Explore (only for PBT method)
            if tuner_cfg.method == "pbt":
                _exploit_explore(
                    population=population,
                    search_space=config.search_space,
                    mode=target.mode,
                    perturbation_factor=tuner_cfg.perturbation_factor,
                )

            # Phase 3: Save state
            if is_rank0():
                save_population_state(population, round_num, state_path)

            # Phase 4: Log round summary
            _log_round_summary(population, round_num, target)

            # Phase 5: Sync before next round
            if is_distributed():
                import torch.distributed

                torch.distributed.barrier()

    except KeyboardInterrupt:
        logger.info("PBT interrupted, saving state...")
        if is_rank0():
            save_population_state(population, round_num, state_path)

    # Finalize
    summary = _finalize(population, config, best_overall, best_config, round_num)
    return summary


def _log_round_summary(
    population: list[PopulationMember],
    round_num: int,
    target: Any,
) -> None:
    """Log a brief summary after each round."""
    metrics_str = ", ".join(f"m{m.member_id}={m.best_metric:.4f}" for m in population)
    logger.info(f"Round {round_num} summary: {metrics_str}")


def _finalize(
    population: list[PopulationMember],
    config: QATSearchConfig,
    best_overall: float,
    best_config: dict | None,
    total_rounds: int,
) -> dict:
    """Export best model and save search summary."""
    target = config.target

    # Find best member (mode-aware)
    best_member = (
        min(population, key=lambda m: m.best_metric)
        if target.mode == "min"
        else max(population, key=lambda m: m.best_metric)
    )

    # Check if target met
    target_met = target.threshold is not None and (
        target.mode == "min"
        and best_member.best_metric <= target.threshold
        or target.mode == "max"
        and best_member.best_metric >= target.threshold
    )

    if target_met:
        logger.info(
            f"Target met! Best {target.metric}={best_member.best_metric:.4f} "
            f"(threshold={target.threshold})"
        )

    # Save summary
    summary = {
        "best_member_id": best_member.member_id,
        "best_config": best_config or best_member.hyperparams,
        "best_metric": best_overall,
        "total_rounds": total_rounds,
        "target_met": target_met,
        "population_size": len(population),
        "all_members": [
            {
                "member_id": m.member_id,
                "best_metric": m.best_metric,
                "total_epochs": m.total_epochs_trained,
                "metric_history": m.metric_history,
                "hyperparams": m.hyperparams,
            }
            for m in population
        ],
    }

    summary_path = os.path.join(config.output_dir, "search_summary.json")
    if is_rank0():
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    logger.info(f"Search summary saved to {summary_path}")
    logger.info(f"Best {target.metric}: {best_overall:.4f}")
    logger.info(f"Target met: {target_met}")

    # Export best model as real-quantized safetensors (rank 0 only)
    export_dir = os.path.join(config.output_dir, "best_model")
    scales_path = os.path.join(best_member.checkpoint_path, "scales.pt")

    if is_rank0() and os.path.exists(scales_path):
        try:
            from .export import export_best_model

            export_best_model(
                model_path=config.model_path,
                tokenizer_path=config.model_path,
                scales_path=scales_path,
                output_dir=export_dir,
                trust_remote_code=config.trust_remote_code,
                weight_format=config.export_weight_format,
            )
            summary["exported"] = True
            summary["export_dir"] = export_dir
            logger.info(f"Best model exported to {export_dir}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            summary["exported"] = False
            summary["export_error"] = str(e)
    elif is_rank0():
        logger.warning(f"No scales.pt found at {scales_path}, skipping export")
        summary["exported"] = False

    # Re-save summary with export status
    if is_rank0():
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return summary

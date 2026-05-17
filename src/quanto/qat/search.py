"""
Ray Tune integration for QAT hyperparameter search.

Builds the Ray search space from YAML config and runs the Tune experiment.
Ray configuration is passthrough — users configure Ray directly via YAML.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from functools import partial

from .config import QATSearchConfig, SearchSpaceDimension
from .trial import qat_trial

logger = logging.getLogger(__name__)


def build_search_space(
    space_config: dict[str, SearchSpaceDimension],
    dataset_ratio_search=None,
) -> dict:
    """
    Convert QATSearchConfig.search_space into a Ray Tune param_space dict.

    For each dimension:
        choices → tune.choice(choices)
        min/max with scale="log" → tune.loguniform(min, max)
        min/max with scale="uniform" → tune.uniform(min, max)
    """
    from ray import tune

    space = {}
    for name, dim in space_config.items():
        if dim.choices is not None:
            space[name] = tune.choice(dim.choices)
        elif dim.min is not None and dim.max is not None:
            if dim.scale == "log":
                space[name] = tune.loguniform(dim.min, dim.max)
            else:
                space[name] = tune.uniform(dim.min, dim.max)

    # Dataset ratio search dimensions
    if dataset_ratio_search is not None:
        n = len(dataset_ratio_search.datasets)
        for i in range(n):
            space[f"ds_ratio_{i}"] = tune.uniform(0.0, 1.0)

    return space


def _dynamic_import(module_path: str, class_name: str):
    """Dynamically import a class from a module path."""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_scheduler(ray_config: dict):
    """Build Ray Tune scheduler from user YAML config."""
    if "scheduler" not in ray_config:
        return None

    sched_cfg = ray_config["scheduler"]
    sched_type = sched_cfg["type"]
    sched_kwargs = {k: v for k, v in sched_cfg.items() if k != "type"}

    sched_cls = _dynamic_import("ray.tune.schedulers", sched_type)
    return sched_cls(**sched_kwargs)


def _build_search_alg(ray_config: dict):
    """Build Ray Tune search algorithm from user YAML config."""
    if "search_alg" not in ray_config:
        return None

    sa_cfg = ray_config["search_alg"]
    sa_type = sa_cfg["type"]
    sa_kwargs = {k: v for k, v in sa_cfg.items() if k != "type"}

    # Search algorithms live in different submodules
    try:
        sa_cls = _dynamic_import("ray.tune.search", sa_type)
    except (ImportError, AttributeError):
        sa_cls = _dynamic_import(f"ray.tune.search.{sa_type.lower()}", sa_type)

    return sa_cls(**sa_kwargs)


def save_search_summary(results, config: QATSearchConfig):
    """Save search summary JSON after all trials complete."""
    best_result = None
    best_config = {}
    best_metrics = {}

    try:
        best_result = results.get_best_result(
            metric=config.target.metric,
            mode=config.target.mode,
        )
        best_config = best_result.config
        best_metrics = best_result.metrics
    except Exception:
        pass

    all_trials = []
    for result in results:
        try:
            trial_data = {
                "trial_id": result.trial_id,
                "config": result.config,
            }
            if result.metrics:
                trial_data.update(
                    {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}
                )
            all_trials.append(trial_data)
        except Exception:
            continue

    threshold = config.target.threshold
    target_met = False
    if threshold is not None and config.target.metric in best_metrics:
        best_val = best_metrics[config.target.metric]
        if config.target.mode == "min" and best_val <= threshold:
            target_met = True
        elif config.target.mode == "max" and best_val >= threshold:
            target_met = True

    summary = {
        "best_trial_id": best_result.trial_id if best_result else None,
        "best_config": best_config,
        "best_metrics": {k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
        "total_trials": len(results),
        "target_met": target_met,
        "all_trials": all_trials,
    }

    summary_path = os.path.join(config.output_dir, "search_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Search summary saved to {summary_path}")
    logger.info(f"Best {config.target.metric}: {best_metrics.get(config.target.metric, 'N/A')}")
    logger.info(f"Target met: {target_met}")

    return summary


def run_qat_search(config: QATSearchConfig, resume: bool = False) -> dict:
    """
    Main entry point. Constructs and runs the Ray Tune experiment.

    Args:
        config: Parsed QATSearchConfig from YAML.
        resume: Whether to resume from a previous experiment.

    Returns:
        Search summary dict.
    """
    from ray.tune import RunConfig, TuneConfig, Tuner

    os.makedirs(config.output_dir, exist_ok=True)

    # Build search space
    param_space = build_search_space(
        config.search_space,
        config.dataset_ratio_search,
    )

    logger.info(f"Search space: {list(param_space.keys())}")
    logger.info(f"Target: {config.target.metric} {config.target.mode} {config.target.threshold}")

    # Build Ray components from user config
    ray_cfg = config.ray_config
    scheduler = _build_scheduler(ray_cfg)
    search_alg = _build_search_alg(ray_cfg)

    # Build trainable
    trainable = partial(qat_trial, search_config=config)

    # Resource constraints
    resources = ray_cfg.get("resources", {})
    per_trial_gpu = resources.get("per_trial_gpu", 1)
    per_trial_cpu = resources.get("per_trial_cpu", 8)

    # Build Tuner
    tuner = Tuner(
        trainable,
        param_space=param_space,
        tune_config=TuneConfig(
            metric=config.target.metric,
            mode=config.target.mode,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=config.target.max_trials,
        ),
        run_config=RunConfig(
            name=ray_cfg.get("name", "qat-search"),
            storage_path=ray_cfg.get("storage", f"file://{config.output_dir}/ray_results"),
            resources_per_trial={
                "gpu": per_trial_gpu,
                "cpu": per_trial_cpu,
            },
        ),
        _tuner_resource_keys=True,
    )

    # Run
    if resume:
        logger.info("Resuming previous experiment")
        tuner = tuner.restore(ray_cfg.get("storage", ""))

    results = tuner.fit()

    # Save summary
    summary = save_search_summary(results, config)

    return summary

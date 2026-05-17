"""
CLI for QAT hyperparameter search.

Usage:
    python -m quanto --qat-search --config qat_search.yaml [--dry-run] [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_search_config
from .search import build_search_space, run_qat_search

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quanto QAT Hyperparameter Search",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to QAT search YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print search space and exit without launching Ray",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous Ray Tune experiment",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    config = load_search_config(args.config)

    if args.dry_run:
        param_space = build_search_space(
            config.search_space,
            config.dataset_ratio_search,
        )
        print("=== QAT Search Config ===")
        print(f"Model: {config.model_path}")
        print(f"Output: {config.output_dir}")
        print(f"Target: {config.target.metric} {config.target.mode} <= {config.target.threshold}")
        print(f"Max trials: {config.target.max_trials}")
        print(f"\nSearch space ({len(param_space)} dimensions):")
        for name, dist in param_space.items():
            print(f"  {name}: {dist}")
        print(f"\nDatasets: {[ds.name for ds in config.train_datasets]}")
        print(f"Ray config: {json.dumps(config.ray_config, indent=2, default=str)}")
        return 0

    # Run search
    summary = run_qat_search(config, resume=args.resume)

    print("\n=== Search Complete ===")
    print(f"Total trials: {summary.get('total_trials', 0)}")
    print(f"Target met: {summary.get('target_met', False)}")
    print(f"Best config: {json.dumps(summary.get('best_config', {}), indent=2, default=str)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

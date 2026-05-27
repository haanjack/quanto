"""
CLI for QAT hyperparameter search.

Usage:
    python -m quanto.qat.cli --config qat_search.yaml [--dry-run] [--resume]
    python -m quanto.qat.cli --config qat_search.yaml --export-best
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import load_search_config
from .sampler import sample_initial_population
from .trial import HFQATTrainer
from .tuner import run_pbt

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
        help="Print population configs and exit without training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous PBT state",
    )
    parser.add_argument(
        "--export-best",
        action="store_true",
        help="Re-export best model from a completed search without re-running",
    )
    parser.add_argument(
        "--search-dir",
        help="Path to completed search output directory (for --export-best)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_search_config(args.config)

    if args.export_best:
        search_dir = args.search_dir or config.output_dir
        summary_path = os.path.join(search_dir, "search_summary.json")
        if not os.path.exists(summary_path):
            print(f"Error: No search_summary.json found in {search_dir}", file=sys.stderr)
            return 1

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        best_member_id = summary["best_member_id"]
        scales_path = os.path.join(
            search_dir, "population", f"member_{best_member_id}", "scales.pt"
        )
        if not os.path.exists(scales_path):
            print(f"Error: No scales.pt at {scales_path}", file=sys.stderr)
            return 1

        from .export import export_best_model

        export_dir = os.path.join(search_dir, "best_model")
        export_best_model(
            model_path=config.model_path,
            tokenizer_path=config.model_path,
            scales_path=scales_path,
            output_dir=export_dir,
            trust_remote_code=config.trust_remote_code,
            weight_format=config.export_weight_format,
        )
        print(f"Exported best model to {export_dir}")
        return 0

    if args.dry_run:
        population = sample_initial_population(
            search_space=config.search_space,
            population_size=config.tuner_config.population_size,
            output_dir=config.output_dir,
            dataset_ratio_search=config.dataset_ratio_search,
        )
        print("=== QAT PBT Search Config ===")
        print(f"Model: {config.model_path}")
        print(f"Method: {config.tuner_config.method}")
        print(f"Population: {config.tuner_config.population_size}")
        print(f"Exploit interval: {config.tuner_config.exploit_interval} epochs")
        print(f"Target: {config.target.metric} {config.target.mode} <= {config.target.threshold}")
        print("\nInitial population:")
        for m in population:
            print(f"  Member {m.member_id}: {json.dumps(m.hyperparams, default=str)}")
        return 0

    # Factory function: creates a fresh HFQATTrainer for each member
    def trainer_factory(search_config):
        return HFQATTrainer(search_config)

    summary = run_pbt(config, trainer_factory, resume=args.resume)

    print("\n=== Search Complete ===")
    print(f"Total rounds: {summary.get('total_rounds', 0)}")
    print(f"Target met: {summary.get('target_met', False)}")
    print(f"Exported: {summary.get('exported', False)}")
    print(f"Best config: {json.dumps(summary.get('best_config', {}), indent=2, default=str)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

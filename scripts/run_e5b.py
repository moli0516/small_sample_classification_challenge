"""Thin CLI: python -m scripts.run_e5b --config configs/e5b_xgboost.json."""
import argparse
from dataclasses import replace
import logging
from pathlib import Path
import sys

from src.config.xgboost_experiment_config import XGBoostExperimentConfig
from src.experiments.xgboost_experiment_runner import XGBoostExperimentRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e5b_xgboost.json")
    parser.add_argument("--output", type=Path, help="Override output_dir; must not already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-reference-alignment", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s | %(message)s")
    config = XGBoostExperimentConfig.from_json(args.config)
    if args.output is not None:
        config = replace(config, output_dir=args.output)
    return XGBoostExperimentRunner().run(config,dry_run=args.dry_run,
        confirm_reference_alignment=args.confirm_reference_alignment)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("E5b interrupted; preserve partial outputs.",file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"E5b stopped: {type(exc).__name__}: {exc}",file=sys.stderr)
        sys.exit(1)

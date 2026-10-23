"""Run one bounded Optuna extension without modifying historical experiment outputs."""
import argparse
from dataclasses import replace
import logging
from pathlib import Path
import sys

from src.config.xgboost_optuna_config import XGBoostOptunaConfig
from src.experiments.xgboost_optuna_runner import XGBoostOptunaRunner


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/e5b_optuna_ext02.json")
    parser.add_argument("--output",type=Path,help="New output directory; do not reuse preflight output.")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--confirm-reference-alignment",action="store_true")
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s | %(message)s")
    config=XGBoostOptunaConfig.from_json(args.config)
    if args.output is not None:
        config=replace(config,output_dir=args.output)
    return XGBoostOptunaRunner().run(config,dry_run=args.dry_run,
        confirm_reference_alignment=args.confirm_reference_alignment)


if __name__=="__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Optuna interrupted. Preserve SQLite/artifacts; no automatic resume.",file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Optuna stopped: {type(exc).__name__}: {exc}",file=sys.stderr)
        sys.exit(1)

"""Thin CLI for fold-local Trial 20 top-k selection."""
import argparse
from dataclasses import replace
import logging
from pathlib import Path
import sys
from src.config.xgboost_feature_selection_config import XGBoostFeatureSelectionConfig
from src.experiments.xgboost_feature_selection_runner import XGBoostFeatureSelectionRunner


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/e5b_feature_selection_ext03.json")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--confirm-reference-alignment",action="store_true")
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s | %(message)s")
    config=XGBoostFeatureSelectionConfig.from_json(args.config)
    if args.output is not None:config=replace(config,output_dir=args.output)
    return XGBoostFeatureSelectionRunner().run(config,dry_run=args.dry_run,
        confirm_reference_alignment=args.confirm_reference_alignment)


if __name__=="__main__":
    try:sys.exit(main())
    except KeyboardInterrupt:
        print("Feature-selection experiment interrupted; preserve partial artifacts.",file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Feature-selection experiment stopped: {type(exc).__name__}: {exc}",file=sys.stderr)
        sys.exit(1)

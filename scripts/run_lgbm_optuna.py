"""Bounded search and export for LightGBM in a fixed three-model ensemble."""
import argparse
import json
import sys
from src.experiments.lgbm_optuna_experiment import LGBMOptunaExperiment


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    search=sub.add_parser("search")
    search.add_argument("--config",default="configs/lgbm_optuna.json")
    search.add_argument("--output")
    search.add_argument("--dry-run",action="store_true")
    pred=sub.add_parser("predict")
    pred.add_argument("--run",default="outputs/results/lgbm_optuna_v1_run_01")
    pred.add_argument("--output",default="outputs/submissions/lgbm_optuna_v1_run_01")
    pred.add_argument("--candidate",default="auto")
    pred.add_argument("--test")
    pred.add_argument("--template")
    pred.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv);runner=LGBMOptunaExperiment()
    if args.action=="search":result=runner.search(args.config,args.output,args.dry_run)
    else:result=runner.predict(args.run,args.output,args.candidate,args.test,args.template,args.dry_run)
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=="__main__":
    try:sys.exit(main())
    except KeyboardInterrupt:print("Interrupted; preserve partial study and choose a new output for any rerun.",file=sys.stderr);sys.exit(130)
    except Exception as exc:print(f"LightGBM Optuna stopped: {type(exc).__name__}: {exc}",file=sys.stderr);sys.exit(1)

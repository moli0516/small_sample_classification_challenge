"""Freeze a development procedure or refit/predict/export; never upload to Kaggle."""
import argparse
import json
import sys
from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="action",required=True)
    freeze=commands.add_parser("freeze")
    freeze.add_argument("--source-run",default="outputs/results/e5b_feature_selection_ext03_run_01")
    freeze.add_argument("--k",type=int,default=20)
    freeze.add_argument("--frozen-dir",default="outputs/frozen/ext03_top20_v1")
    freeze.add_argument("--confirm-reference-alignment",action="store_true")
    predict=commands.add_parser("predict")
    predict.add_argument("--frozen-dir",default="outputs/frozen/ext03_top20_v1")
    predict.add_argument("--output",required=True)
    predict.add_argument("--test",default=None)
    predict.add_argument("--template",default=None)
    predict.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    service=FrozenXGBoostSubmission()
    if args.action=="freeze":
        result=service.freeze(args.source_run,args.k,args.frozen_dir,args.confirm_reference_alignment)
        print("Candidate procedure frozen:",result)
        print("No models trained and no test predictions generated.")
        return 0
    result=service.refit_predict(args.frozen_dir,args.output,args.test,args.template,args.dry_run)
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    try:sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; do not upload partial outputs.",file=sys.stderr);sys.exit(130)
    except Exception as exc:
        print(f"Freeze/submission stopped: {type(exc).__name__}: {exc}",file=sys.stderr);sys.exit(1)

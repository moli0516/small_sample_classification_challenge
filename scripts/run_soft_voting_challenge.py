"""Fixed 50/50 L1 and frozen top-k XGBoost soft-voting experiment."""
import argparse
import json
import sys
from src.experiments.soft_voting_challenge import SoftVotingChallenge


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="action",required=True)
    cv=commands.add_parser("cv")
    cv.add_argument("--frozen-dir",default="outputs/frozen/ext03_top20_v1")
    cv.add_argument("--l1-config",default="configs/e4_l1_c_1.json")
    cv.add_argument("--output",required=True)
    cv.add_argument("--dry-run",action="store_true")
    predict=commands.add_parser("predict")
    predict.add_argument("--cv-run",required=True)
    predict.add_argument("--output",required=True)
    predict.add_argument("--test",default=None)
    predict.add_argument("--template",default=None)
    predict.add_argument("--dry-run",action="store_true")
    args=parser.parse_args();runner=SoftVotingChallenge()
    if args.action=="cv":result=runner.cv(args.frozen_dir,args.l1_config,args.output,args.dry_run)
    else:result=runner.predict(args.cv_run,args.output,args.test,args.template,args.dry_run)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    try:main()
    except KeyboardInterrupt:print("Interrupted; do not upload partial outputs.",file=sys.stderr);sys.exit(130)
    except Exception as exc:print(f"Soft voting stopped: {type(exc).__name__}: {exc}",file=sys.stderr);sys.exit(1)

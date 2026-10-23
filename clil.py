"""Unified experiment CLI. Name clil.py is intentional, as requested."""
import argparse
from importlib import import_module
from importlib.metadata import version,PackageNotFoundError
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
ROUTES={
    "audit":"scripts.audit_data", "baseline":"scripts.run_baseline",
    "e5":"scripts.run_e5", "xgb":"scripts.run_e5b",
    "xgb-optuna":"scripts.run_e5b_optuna", "xgb-features":"scripts.run_e5b_feature_selection",
    "frozen-xgb":"scripts.run_frozen_xgboost_submission", "soft-vote":"scripts.run_soft_voting_challenge",
    "registry":"scripts.run_registered_ensembles", "extension":"scripts.run_extension",
    "lgbm-optuna":"scripts.run_lgbm_optuna",
}
LEGACY=["audit_data","check_e5_warning","create_e2_configs","create_e3_configs","create_e4_configs",
    "make_submission","run_baseline","run_e5","run_e5b","run_e5b_feature_selection","run_e5b_optuna",
    "run_extension","run_frozen_xgboost_submission","run_registered_ensembles","run_soft_voting_challenge","run_lgbm_optuna"]


def execute(arguments):
    return subprocess.run([sys.executable,*arguments],cwd=ROOT,check=False).returncode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name,module in ROUTES.items():
        sub.add_parser(name,help=f"Forward arguments unchanged to {module}",add_help=False)
    sub.add_parser("models",help="List registered model types")
    doctor=sub.add_parser("doctor",help="Check package metadata and load native libraries")
    doctor.add_argument("--metadata-only",action="store_true")
    tests=sub.add_parser("test",help="Run pytest using the current interpreter")
    tests.add_argument("--scope",choices=["new","registry","all"],default="all")
    report=sub.add_parser("results",help="Read saved reports, never start training")
    report.add_argument("--run",required=True)
    old=sub.add_parser("run",help="Forward to an existing project script",add_help=False)
    old.add_argument("script",choices=LEGACY)
    sub.add_parser("plan",help="Print the safe current experiment command sequence")
    args,remaining=parser.parse_known_args(argv)
    if args.command in ROUTES:return execute(["-m",ROUTES[args.command],*remaining])
    if args.command=="run":return execute(["-m",f"scripts.{args.script}",*remaining])
    if args.command=="models":
        if remaining:parser.error("models accepts no extra arguments")
        return execute(["-m","scripts.run_registered_ensembles","list"])
    if args.command=="test":
        paths={"new":["tests/test_lgbm_optuna_experiment.py","tests/test_clil.py"],
               "registry":["tests/test_registered_ensembles.py"],"all":[]}[args.scope]
        return execute(["-m","pytest","-q",*paths,*remaining])
    if remaining:parser.error(f"Unknown arguments: {' '.join(remaining)}")
    if args.command=="doctor":
        result={"python":sys.version.split()[0],"project_root":str(ROOT),"packages":{}};ok=True
        for package,module in [("numpy","numpy"),("pandas","pandas"),("scikit-learn","sklearn"),
                               ("joblib","joblib"),("pytest","pytest"),("xgboost","xgboost"),("lightgbm","lightgbm"),("optuna","optuna")]:
            try:
                entry={"version":version(package)}
                if not args.metadata_only:import_module(module);entry["import_ok"]=True
            except Exception as exc:entry={"error":f"{type(exc).__name__}: {exc}"};ok=False
            result["packages"][package]=entry
        print(json.dumps(result,indent=2));return 0 if ok else 1
    if args.command=="results":
        path=(ROOT/Path(args.run).expanduser()).resolve()
        if not path.is_dir():parser.error("Run directory does not exist")
        shown=False
        for name in ("run_manifest.json","decision.json","best_trial.json","summary.json","prediction_manifest.json"):
            if (path/name).is_file():print(f"\n=== {name} ===\n{(path/name).read_text()}");shown=True
        for name in ("comparison.csv","trials.csv"):
            if (path/name).is_file():print(f"\n=== {name} ===\n{(path/name).read_text()}");shown=True
        return 0 if shown else 1
    if args.command=="plan":
        print("""python -m pip install -r requirements-lgbm-optuna.txt
python clil.py doctor
python clil.py test --scope new
python clil.py test
python clil.py lgbm-optuna search --dry-run
python clil.py lgbm-optuna search
python clil.py results --run outputs/results/lgbm_optuna_v1_run_01
# ONLY if decision.json has promote=true:
python clil.py lgbm-optuna predict --dry-run
python clil.py lgbm-optuna predict
# Review prediction_manifest.json before manually uploading submission.csv.""")
        return 0
    return 1


if __name__=="__main__":
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)

"""Single entry point: list registered models, compare, or export the selected blend."""
import argparse
import json
from pathlib import Path
import sys
from src.modeling.model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    cv = commands.add_parser("cv")
    cv.add_argument("--config", default="configs/registry_ensembles.json")
    cv.add_argument("--output")
    cv.add_argument("--no-cache", action="store_true")
    cv.add_argument("--dry-run", action="store_true")
    pred = commands.add_parser("predict")
    pred.add_argument("--run", default="outputs/results/registry_ensembles_v1_run_01")
    pred.add_argument("--output", default="outputs/submissions/registry_ensemble_v1_run_01")
    pred.add_argument("--candidate", default="auto")
    pred.add_argument("--test")
    pred.add_argument("--template")
    pred.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(ModelRegistry.names())); return
    from src.experiments.registered_ensembles import RegisteredEnsembles
    runner = RegisteredEnsembles()
    if args.command == "cv":
        result = runner.cv(args.config, args.output, args.no_cache, args.dry_run)
    else:
        result = runner.predict(args.run, args.output, args.candidate, args.test, args.template, args.dry_run)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; do not upload partial outputs.", file=sys.stderr); sys.exit(130)
    except Exception as exc:
        print(f"Registry experiment stopped: {type(exc).__name__}: {exc}", file=sys.stderr); sys.exit(1)

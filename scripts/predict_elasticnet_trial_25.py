"""Predict/export the frozen Elastic-net Trial25; never upload to Kaggle."""
import argparse
import json
import sys
from src.submission.elasticnet_trial_submission import ElasticnetTrialSubmission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/results/elasticnet_optuna_v1_run_01")
    parser.add_argument("--config", default="configs/elasticnet_optuna_v1.json")
    parser.add_argument("--output", default="outputs/submissions/elasticnet_optuna_trial_25_run_01")
    parser.add_argument("--test")
    parser.add_argument("--template")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = ElasticnetTrialSubmission().predict(**vars(args))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; preserve outputs and use a fresh output directory.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Trial25 export stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

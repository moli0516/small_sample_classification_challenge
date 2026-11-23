"""Bounded automatic RF max_features tuning; no test prediction or upload."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rf_auto_v2.json")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from src.experiments.rf_auto_search import RFAutoSearch
    if not args.dry_run:
        print("Verifying historical sources and cached RF control before fitting...", flush=True)
    print(json.dumps(RFAutoSearch().run(args.config, args.output, args.dry_run), indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"RF auto search stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

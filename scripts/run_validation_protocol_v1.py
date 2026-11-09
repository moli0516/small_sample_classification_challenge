"""Run the fixed validation protocol extension without touching RF or submissions."""
import argparse
import json
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/validation_protocol_v1.json")
    p.add_argument("--output")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    from src.extensions.validation_protocol_v1 import ValidationProtocolV1
    if not args.dry_run:print("Verifying frozen sources before generating fresh validation fits...",flush=True)
    print(json.dumps(ValidationProtocolV1().run(args.config,args.output,args.dry_run),indent=2,allow_nan=False))


if __name__=="__main__":
    try:main()
    except KeyboardInterrupt:sys.exit(130)
    except Exception as exc:
        print(f"Validation study stopped: {type(exc).__name__}: {exc}",file=sys.stderr);sys.exit(1)

"""Predict the Trial21 standalone submission from the frozen model bundle; no fits.

Loads the frozen bundle written by the original Trial21 export, scores the
competition test file, aligns the probabilities to the submission template and
writes `<output-dir>/standalone/submission.csv`. Nothing is retrained and no
source-provenance gate is involved.

The full export CLI (scripts.export_generative_trial21_v1) only passes its
source-provenance check at the original absolute path with the original source
versions; see "Known reproducibility limitations" in README.md.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = "outputs/submissions/generative_optuna_trial21_v1/model_bundle.joblib"
DEFAULT_BUNDLE_SHA256 = "4ef86a6b5d72177a1f60f954ee2177e3a20a0b1f082120b0ec9c6406b989e212"
DEFAULT_TEST = "data/raw/test.csv"
DEFAULT_TEMPLATE = "data/raw/sample_submission.csv"
DEFAULT_OUTPUT_DIR = "outputs/repro_check/trial21_bundle"
COLUMNS = ["id", "target"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def display(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def csv_frame(path, columns=None):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(set(header)) != len(header):
        raise ValueError(f"Invalid CSV header in {path}: {header}")
    if columns is not None and header != columns:
        raise ValueError(f"Unexpected columns in {path}: {header}")
    frame = pd.read_csv(path, dtype={"id": "string"}, float_precision="round_trip")
    if frame.id.isna().any() or not frame.id.is_unique or frame.id.str.strip().eq("").any():
        raise ValueError(f"Missing, empty or duplicate ids in {path}")
    return frame


def predict(bundle, features):
    """Mean of the bundle's per-seed GMM probabilities in the bundle's feature order."""
    if list(features.columns) != list(bundle["features"]):
        raise ValueError("Test feature order differs from the frozen bundle")
    z = bundle["projector"].transform(features)
    probabilities = np.mean([model.predict_proba(z)[:, 1] for model in bundle["models"]], axis=0)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities outside [0, 1]")
    return probabilities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE, help="frozen Trial21 model bundle")
    parser.add_argument("--test", default=DEFAULT_TEST, help="competition test CSV")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="sample submission defining the id order")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="fresh directory; never overwritten")
    parser.add_argument("--expected-bundle-sha256", default=DEFAULT_BUNDLE_SHA256,
                        help="required bundle SHA256")
    args = parser.parse_args(argv)

    bundle_path = resolve(args.bundle)
    test_path = resolve(args.test)
    template_path = resolve(args.template)
    output_dir = resolve(args.output_dir)
    for label, path in [("--bundle", bundle_path), ("--test", test_path), ("--template", template_path)]:
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if output_dir.exists():
        parser.error(f"Refusing to write into an existing output directory: {display(output_dir)}")

    bundle_sha256 = digest(bundle_path)
    if bundle_sha256 != args.expected_bundle_sha256:
        print(f"Bundle SHA256 mismatch: {display(bundle_path)} is {bundle_sha256}, "
              f"expected {args.expected_bundle_sha256}. Refusing to predict from a changed bundle.",
              file=sys.stderr)
        raise SystemExit(1)

    bundle = joblib.load(bundle_path)
    test = csv_frame(test_path)
    template = csv_frame(template_path, COLUMNS)
    if len(test) != len(template):
        parser.error(f"Test and template row counts differ: {len(test)} != {len(template)}")
    if set(test.id) != set(template.id):
        parser.error("Test and template cover different ids")
    if set(test.columns) != {"id", *bundle["features"]}:
        parser.error(f"Test columns differ from id plus the bundle features: {list(test.columns)}")

    with threadpool_limits(limits=1):
        raw = predict(bundle, test[bundle["features"]])
    # Template order, not test order, defines the submission row order.
    ordered = pd.Series(raw, index=test.id).loc[template.id].to_numpy()

    directory = output_dir / "standalone"
    directory.mkdir(parents=True)
    output = directory / "submission.csv"
    pending = output.with_name("submission.pending.csv")
    pd.DataFrame({"id": template.id.to_numpy(), "target": ordered}).to_csv(
        pending, index=False, float_format="%.17g", lineterminator="\n")
    reread = csv_frame(pending, COLUMNS)
    if not np.array_equal(reread.id.to_numpy(), template.id.to_numpy()) or not np.array_equal(
            reread.target.to_numpy(dtype=float), ordered):
        pending.unlink(missing_ok=True)
        parser.error("Round-trip check failed: ids or probabilities changed")
    pending.replace(output)

    result = {
        "status": "ok",
        "fits_performed": 0,
        "bundle": {"path": display(bundle_path), "sha256": bundle_sha256},
        "rows": len(ordered),
        "output": {"path": str(output), "sha256": digest(output)},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

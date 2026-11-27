"""Replay submission A095 from frozen exports; no fits, no model loading.

A095 = alpha*G_new + (1-alpha)*H_old with alpha = 0.95, where
  G_new = generative_optuna_trial21_v1/standalone/submission.csv
  H_old = generative_blend_gmm05_v1/submission.csv (0.5*K8 GMM + 0.5*Trial25)

This replays only the blend. Refitting G_new or H_old goes through the source
export scripts and their provenance checks (see README reproducibility notes).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.95
MANIFEST = ROOT / "outputs/submissions/trial21_A095_manual_v1/prediction_manifest.json"
DEFAULT_G_NEW = "outputs/submissions/generative_optuna_trial21_v1/standalone/submission.csv"
DEFAULT_H_OLD = "outputs/submissions/generative_blend_gmm05_v1/submission.csv"
DEFAULT_TEMPLATE = "data/raw/sample_submission.csv"
DEFAULT_OUTPUT = "outputs/repro_check/a095/submission.csv"
DEFAULT_SHA256 = "d772108c1815ccd7cef1018995c1d8873a42a87fedf6d47ab751971e22b991e9"
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


def recorded_sha256(manifest, path):
    """Recorded hash for a repo-relative input, located by path suffix in the A095 manifest."""
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError(f"Input is outside the repository, so the A095 manifest cannot verify it: {path}")
    matches = {key: value for key, value in manifest["source_input_and_code_sha256"].items()
               if key.replace("\\", "/").endswith("/" + relative)}
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest entry for {relative}, found {len(matches)}")
    return next(iter(matches.values()))


def read_submission(path):
    frame = pd.read_csv(path, dtype={"id": "string"}, float_precision="round_trip")
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    if frame.id.isna().any() or not frame.id.is_unique or frame.id.str.len().eq(0).any():
        raise ValueError(f"Missing, empty or duplicate ids in {path}")
    probabilities = frame.target.to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"Probabilities outside [0, 1] in {path}")
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--g-new", default=DEFAULT_G_NEW, help="frozen Trial21 standalone submission")
    parser.add_argument("--h-old", default=DEFAULT_H_OLD, help="frozen 0.5*K8 GMM + 0.5*Trial25 submission")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE,
                        help="optional id-order template; checked only when the file exists")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="fresh output CSV; never overwritten")
    parser.add_argument("--expected-sha256", default=DEFAULT_SHA256, help="required output SHA256")
    args = parser.parse_args(argv)

    g_new, h_old = resolve(args.g_new), resolve(args.h_old)
    output = resolve(args.output)
    template = resolve(args.template)
    for label, path in [("--g-new", g_new), ("--h-old", h_old)]:
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    for path in [output, output.with_name(output.name + ".pending")]:
        if path.exists():
            parser.error(f"Refusing to overwrite existing path: {path}")

    manifest = json.loads(MANIFEST.read_text())
    inputs = {}
    for label, path in [("g_new", g_new), ("h_old", h_old)]:
        current = digest(path)
        recorded = recorded_sha256(manifest, path)
        if current != recorded:
            parser.error(f"{label} SHA256 differs from A095 manifest: {current} != {recorded}")
        inputs[label] = {"path": display(path), "sha256": current}

    g = read_submission(g_new)
    h = read_submission(h_old)
    if set(g.id) != set(h.id):
        parser.error("G_new and H_old cover different ids")
    h = h.set_index("id").loc[g.id].reset_index(drop=True)
    if template.is_file():
        template_ids = read_submission(template).id.reset_index(drop=True)
        if not template_ids.equals(g.id.reset_index(drop=True)):
            parser.error(f"Id order differs from template: {template}")

    probabilities = ALPHA * g.target.to_numpy(dtype=float) + (1.0 - ALPHA) * h.target.to_numpy(dtype=float)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".pending")
    pd.DataFrame({"id": g.id, "target": probabilities}).to_csv(
        pending, index=False, float_format="%.17g", lineterminator="\n")
    reread = read_submission(pending)
    if not reread.id.equals(g.id) or not np.array_equal(reread.target.to_numpy(dtype=float), probabilities):
        pending.unlink(missing_ok=True)
        parser.error("Round-trip check failed: ids or probabilities changed")
    pending.replace(output)

    result = {
        "status": "ok",
        "formula": "0.95*G_new + 0.05*H_old",
        "rows": len(probabilities),
        "inputs": inputs,
        "output": {"path": display(output), "sha256": digest(output)},
        "expected_sha256": args.expected_sha256,
        "expected_from_manifest": manifest.get("submission_sha256"),
    }
    print(json.dumps(result, indent=2))
    if result["output"]["sha256"] != args.expected_sha256:
        print(f"SHA256 mismatch: {result['output']['sha256']} != {args.expected_sha256}")
        raise SystemExit(1)
    print("SHA256 matches expected value")


if __name__ == "__main__":
    main()

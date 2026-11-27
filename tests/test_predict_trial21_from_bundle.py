import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/predict_trial21_from_bundle.py"
FEATURES = [f"f{i}" for i in range(6)]


@pytest.fixture
def synthetic(tmp_path):
    """A tiny two-class bundle shaped like the frozen one, built from synthetic data only."""
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(12, len(FEATURES))), columns=FEATURES)
    y = np.array([0, 1] * 6)
    projector = PCA(n_components=2, random_state=0).fit(x)
    z = projector.transform(x)
    bundle = dict(projector=projector, models=[LogisticRegression().fit(z, y) for _ in range(3)],
                  features=FEATURES, parameters={}, seeds=[1, 2, 3], selected_trial=21, mode="training_only")
    bundle_path = tmp_path / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    ids = [f"{i:05d}" for i in range(12)]
    test_path = tmp_path / "test.csv"
    x.assign(id=ids)[["id", *FEATURES]].to_csv(test_path, index=False)
    template_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": list(reversed(ids)), "target": 0.0}).to_csv(template_path, index=False)
    return dict(bundle=bundle_path, test=test_path, template=template_path,
                sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest())


def run(synthetic, output_dir, expected=None):
    args = [sys.executable, str(SCRIPT), "--bundle", str(synthetic["bundle"]),
            "--test", str(synthetic["test"]), "--template", str(synthetic["template"]),
            "--output-dir", str(output_dir),
            "--expected-bundle-sha256", expected or synthetic["sha256"]]
    return subprocess.run(args, capture_output=True, text=True)


def test_writes_id_and_target_in_template_order(synthetic, tmp_path):
    output_dir = tmp_path / "fresh"
    result = run(synthetic, output_dir)
    assert result.returncode == 0, result.stderr

    output = output_dir / "standalone" / "submission.csv"
    frame = pd.read_csv(output, dtype={"id": "string"})
    assert list(frame.columns) == ["id", "target"]
    template_ids = pd.read_csv(synthetic["template"], dtype={"id": "string"}).id
    assert frame.id.tolist() == template_ids.tolist()
    assert frame.target.between(0, 1).all()

    # Probabilities must follow their own id through the reordering, not the test row order.
    bundle = joblib.load(synthetic["bundle"])
    test = pd.read_csv(synthetic["test"], dtype={"id": "string"})
    z = bundle["projector"].transform(test[FEATURES])
    expected = np.mean([m.predict_proba(z)[:, 1] for m in bundle["models"]], axis=0)
    by_id = dict(zip(test.id, expected))
    assert np.allclose(frame.target.to_numpy(), [by_id[i] for i in frame.id], rtol=0, atol=1e-12)

    reported = json.loads(result.stdout)
    assert reported["fits_performed"] == 0
    assert reported["rows"] == len(template_ids)
    assert reported["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_refuses_existing_output_directory(synthetic, tmp_path):
    output_dir = tmp_path / "taken"
    output_dir.mkdir()
    result = run(synthetic, output_dir)
    assert result.returncode != 0
    assert "existing output directory" in result.stderr
    assert not (output_dir / "standalone").exists()


def test_refuses_bundle_with_unexpected_sha256(synthetic, tmp_path):
    output_dir = tmp_path / "fresh"
    result = run(synthetic, output_dir, expected="0" * 64)
    assert result.returncode != 0
    assert "Bundle SHA256 mismatch" in result.stderr
    assert not output_dir.exists()

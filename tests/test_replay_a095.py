import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
G_NEW = ROOT / "outputs/submissions/generative_optuna_trial21_v1/standalone/submission.csv"
H_OLD = ROOT / "outputs/submissions/generative_blend_gmm05_v1/submission.csv"
MANIFEST = ROOT / "outputs/submissions/trial21_A095_manual_v1/prediction_manifest.json"
EXPECTED = "d772108c1815ccd7cef1018995c1d8873a42a87fedf6d47ab751971e22b991e9"
MISSING = ", ".join(str(p.relative_to(ROOT)) for p in (G_NEW, H_OLD, MANIFEST) if not p.is_file())


def replay(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/replay_a095.py"), *args], capture_output=True, text=True)


@pytest.mark.skipif(bool(MISSING), reason="frozen A095 inputs absent: " + MISSING)
def test_replay_a095(tmp_path):
    output = tmp_path / "submission.csv"
    result = replay("--output", str(output))
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256(output.read_bytes()).hexdigest() == EXPECTED

    before = output.read_bytes()
    assert replay("--output", str(output)).returncode != 0
    assert output.read_bytes() == before

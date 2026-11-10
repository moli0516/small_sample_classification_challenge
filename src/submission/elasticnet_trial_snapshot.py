"""Read-only verification of the completed, selected Elastic-net Trial25."""
from importlib.metadata import version
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class ElasticnetTrialSnapshot:
    tolerance = 1e-12
    expected_auc = 0.9092096430139908
    control_auc = 0.9084647196603719
    expected_parameters = dict(C=0.18702703609689042,
                               l1_ratio=0.9589771634360093,
                               alpha=0.06655791394873416)

    def __init__(self, helper=None, prepare_study=None, version_reader=version):
        if helper is None:
            from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
            helper = FrozenXGBoostSubmission()
        self.helper = helper
        self.prepare_study = prepare_study
        self.version_reader = version_reader

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def verify_hashes(self, hashes):
        for path, digest in hashes.items():
            if self.helper._sha(path) != digest:
                raise ValueError("Input/code/artifact changed: " + str(path))

    def artifacts(self, run):
        manifest = self.read(run / "run_manifest.json")
        if (manifest.get("status") != "completed" or manifest.get("planned_trials") != 30
                or manifest.get("completed_trials") != 30 or manifest.get("planned_fits") != 1500
                or manifest.get("completed_fits") != 1500
                or manifest.get("test_opened") is not False
                or manifest.get("kaggle_upload_performed") is not False):
            raise ValueError("Expected original completed 30-trial/1500-fit no-test study")
        checks = manifest.get("artifact_sha256", {})
        required = {"settings.json", "decision.json", "best_trial.json", "trials.json", "trials.csv",
                    "cohort.csv", "split_indices.json", "trials/trial_0000/result.json",
                    "trials/trial_0000/predictions.npz", "trials/trial_0025/result.json",
                    "trials/trial_0025/parameters.json", "trials/trial_0025/predictions.npz"}
        if not required <= checks.keys():
            raise ValueError("Study artifact hashes incomplete")
        hashes = {str(run / "run_manifest.json"): self.helper._sha(run / "run_manifest.json")}
        for name, digest in checks.items():
            path = (run / name).resolve()
            if not path.is_relative_to(run) or not path.is_file():
                raise ValueError("Unsafe/missing study artifact: " + name)
            hashes[str(path)] = digest
        self.verify_hashes(hashes)
        return hashes

    @staticmethod
    def probability(values, size):
        p = np.asarray(values, dtype=float)
        if p.shape != (size,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probability vector")
        return p

    def score(self, context, values):
        scores = []
        for split in context["groups"]:
            sl = slice(split["start"], split["stop"])
            scores.append(dict(repeat=split["repeat"], fold=split["fold"],
                               auc=roc_auc_score(context["frame"].target.iloc[sl], values[sl])))
        frame = pd.DataFrame(scores)
        if len(frame) != 50 or frame.duplicated(["repeat", "fold"]).any():
            raise ValueError("Expected50 unique folds")
        for repeat in range(10):
            if set(frame.loc[frame.repeat == repeat, "fold"]) != set(range(5)):
                raise ValueError("Expected10 complete repetitions")
        return frame.groupby("repeat").auc.mean().sort_index()

    def close(self, actual, expected, label):
        if not np.isclose(actual, expected, rtol=0, atol=self.tolerance):
            raise ValueError("Score mismatch: " + label)

    def compare(self, current, reference, recorded):
        delta = current - reference
        actual = dict(mean_difference=float(delta.mean()), difference_sd=float(delta.std(ddof=1)),
                      wins=int((delta > self.tolerance).sum()),
                      losses=int((delta < -self.tolerance).sum()),
                      ties=int((abs(delta) <= self.tolerance).sum()))
        for key, value in actual.items():
            self.close(value, recorded[key], key)
        if actual["mean_difference"] <= self.tolerance or actual["wins"] < 6:
            raise ValueError("Selected trial did not pass both-control screening")

    def load(self, run_path, config_path, output):
        h = self.helper
        run, dest = h._path(run_path), h._path(output)
        if dest.exists():
            raise FileExistsError("Fresh prediction output required; no overwrite/resume")
        if dest.is_relative_to(run) or run.is_relative_to(dest):
            raise ValueError("Output overlaps source study")
        hashes = self.artifacts(run)
        settings, decision, best = (self.read(run / name) for name in
                                   ("settings.json", "decision.json", "best_trial.json"))
        if (decision.get("selected_trial") != 25 or best.get("number") != 25
                or best.get("state") != "COMPLETE"
                or decision.get("control_reproduced") is not True
                or decision.get("screening_passed") is not True
                or decision.get("screening_both_controls") is not True
                or decision.get("automatic_promotion") is not False
                or best.get("parameters") != self.expected_parameters
                or decision.get("parameters") != best["parameters"]):
            raise ValueError("Expected frozen, both-controls-passed Trial25")
        self.close(best["objective_auc"], self.expected_auc, "Trial25 identity")
        self.close(decision["selected_mean_auc"], best["objective_auc"], "decision")
        selected = run / "trials/trial_0025"
        if self.read(selected / "result.json") != best or self.read(selected / "parameters.json") != best["parameters"]:
            raise ValueError("Selected result/parameters disagree")
        trials = self.read(run / "trials.json")
        if (len(trials) != 30 or [t["number"] for t in trials] != list(range(30))
                or any(t["state"] != "COMPLETE" or not np.isfinite(t["value"]) for t in trials)):
            raise ValueError("Incomplete trial accounting")
        for trial in trials:
            p = trial["parameters"]
            if (set(p) != {"C", "l1_ratio", "alpha"} or not .001 <= p["C"] <= 1.
                    or not .5 <= p["l1_ratio"] <= 1. or not .01 <= p["alpha"] <= .2):
                raise ValueError("Trial outside declared space")
        winner = min(trials, key=lambda t: (-t["value"], t["number"]))
        if winner["number"] != 25 or winner["parameters"] != best["parameters"]:
            raise ValueError("Winner rule does not select Trial25")
        self.close(winner["value"], best["objective_auc"], "trials.json")
        table = pd.read_csv(run / "trials.csv", float_precision="round_trip")
        if len(table) != 30 or table.number.tolist() != list(range(30)):
            raise ValueError("Invalid trials.csv")
        for row, trial in zip(table.to_dict("records"), trials):
            if row["state"] != trial["state"]:
                raise ValueError("Trial states disagree")
            self.close(row["objective_auc"], trial["value"], "trial CSV objective")
            for key in ("C", "l1_ratio", "alpha"):
                self.close(row[key], trial["parameters"][key], "trial CSV parameter")
        if trials[0]["parameters"] != dict(C=.1, l1_ratio=.75, alpha=.05):
            raise ValueError("Incorrect control parameters")
        if trials[0]["user_attrs"].get("control_reproduced") is not True:
            raise ValueError("Missing control reproduction")
        control = self.read(run / "trials/trial_0000/result.json")
        if control.get("control_reproduced") is not True or control.get("parameters") != trials[0]["parameters"]:
            raise ValueError("Invalid control result")
        self.close(control["objective_auc"], self.control_auc, "control identity")
        self.close(trials[0]["value"], self.control_auc, "control trial value")
        if self.prepare_study is None:
            # Read-only compatibility adapter: historical hashed scripts MUST stay unchanged.
            from scripts.run_elasticnet_optuna_v1 import prepare
        else:
            prepare = self.prepare_study
        c = prepare(str(h._path(config_path)), output=str(dest))
        if (c["optuna_config"] != settings["config"]
                or c["elasticnet_hashes"] != settings["input_and_code_sha256"]):
            raise ValueError("Study inputs/config/code changed")
        for package, expected in settings["versions"].items():
            if self.version_reader(package) != expected:
                raise ValueError("Study package version changed: " + package)
        if (settings.get("solver"), settings.get("max_iter"), settings.get("tol"), settings.get("model_seed")) != ("saga", 20000, 1e-4, 42):
            raise ValueError("Unexpected evaluated Elastic-net procedure")
        cohort = pd.read_csv(run / "cohort.csv", dtype={"id": "string"}, float_precision="round_trip")
        expected_frame = c["frame"].copy()
        expected_frame["id"] = expected_frame["id"].astype("string")
        if not cohort.equals(expected_frame) or self.read(run / "split_indices.json") != c["groups"]:
            raise ValueError("Cohort or ordered split identity changed")
        size = len(c["frame"])
        for number, expected_auc in ((0, self.control_auc), (25, best["objective_auc"])):
            with np.load(run / "trials" / f"trial_{number:04d}" / "predictions.npz", allow_pickle=False) as saved:
                raw = self.probability(saved["elasticnet"], size)
                p = self.probability(saved["ensemble"], size)
            alpha = trials[number]["parameters"]["alpha"]
            reconstructed = (1 - alpha) * c["baseline"] + alpha * raw
            if not np.allclose(p, reconstructed, rtol=0, atol=self.tolerance):
                raise ValueError("Saved ensemble is not the evaluated formula")
            repeats = self.score(c, reconstructed)
            self.close(repeats.mean(), expected_auc, "recomputed ensemble")
            if number == 25:
                self.close(repeats.std(ddof=1), best["repeat_sd"], "repeat SD")
                self.close(self.score(c, raw).mean(), best["standalone_auc"], "standalone")
                reference = self.score(c, c["baseline"])
                catboost = self.score(c, c["catboost_control"])
                self.compare(repeats, reference, best["reference_comparison"])
                self.compare(repeats, catboost, best["catboost_comparison"])
                if (decision["reference_comparison"] != best["reference_comparison"]
                        or decision["catboost_control_comparison"] != best["catboost_comparison"]):
                    raise ValueError("Decision comparisons disagree")
        hashes.update(c["elasticnet_hashes"])
        if any(Path(path).resolve().is_relative_to(dest) for path in hashes):
            raise ValueError("Output overlaps watched inputs")
        self.verify_hashes(hashes)
        return dict(run=run, output=dest, context=c, parameters=best["parameters"],
                    best=best, decision=decision, hashes=hashes, settings=settings)

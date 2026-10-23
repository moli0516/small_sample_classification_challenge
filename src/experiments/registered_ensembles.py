"""One runner: registry models, verified OOF reuse, equal-weight comparisons/export."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import re
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.modeling.model_registry import ModelRegistry, RegisteredModel, soft_vote


class RegisteredEnsembles:
    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def validate(config):
        required = {"frozen_dir", "l1_config", "reuse_cv_run", "output_dir", "models", "ensembles", "baseline"}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(f"Configuration requires exactly: {sorted(required)}")
        models, ensembles = config["models"], config["ensembles"]
        if not isinstance(models, dict) or not {"l1", "xgboost"} <= models.keys():
            raise ValueError("Keep l1/xgboost aliases for the established controls.")
        if not isinstance(ensembles, dict) or config["baseline"] not in ensembles:
            raise ValueError("Baseline ensemble is missing.")
        for name in [*models, *ensembles]:
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError("Names must be safe identifiers.")
        if set(models) & set(ensembles):
            raise ValueError("Model and ensemble names must be distinct.")
        for spec in models.values():
            if not isinstance(spec, dict) or not {"type"} <= spec.keys() or set(spec) - {"type", "parameters"}:
                raise ValueError("Model spec requires type and optional parameters.")
            if spec["type"] not in ModelRegistry.names() or not isinstance(spec.get("parameters", {}), dict):
                raise ValueError("Unknown registered type or invalid parameters.")
        if models["l1"] != {"type": "l1_pipeline"} or models["xgboost"] != {"type": "xgb_topk"}:
            raise ValueError("Do not change the established l1/xgboost control definitions.")
        for spec in ensembles.values():
            if not isinstance(spec, dict) or set(spec) != {"members", "weights"}:
                raise ValueError("Ensemble requires members/weights.")
            soft_vote({name: np.array([.5]) for name in models}, spec["members"], spec["weights"])
        base = ensembles[config["baseline"]]
        if base["members"] != ["l1", "xgboost"] or not np.allclose(np.array(base["weights"], float) / max(base["weights"]) / (sum(w / max(base["weights"]) for w in base["weights"])), [.5, .5]):
            raise ValueError("Baseline must remain equal-weight l1+xgboost.")
        return config

    @staticmethod
    def assemble(probabilities, ensembles):
        result = dict(probabilities)
        for name, spec in ensembles.items():
            result[name] = soft_vote(probabilities, spec["members"], spec["weights"])
        return result

    def _cache(self, c, run):
        h = self.helper
        manifest = self.read(run / "run_manifest.json")
        if manifest.get("status") != "completed" or manifest.get("completed_folds") != 50 or manifest.get("control_auc_reproduced") is not True:
            raise ValueError("OOF cache needs a completed, control-reproduced 50-fold study.")
        required = {"config.json", "summary.json", "fold_scores.csv", "validation_probabilities.csv"}
        if not required <= manifest.get("artifact_sha256", {}).keys():
            raise ValueError("OOF artifact checksums are incomplete.")
        paths = [run / "run_manifest.json"]
        for name, digest in manifest["artifact_sha256"].items():
            path = (run / name).resolve()
            if not path.is_relative_to(run) or h._sha(path) != digest:
                raise ValueError(f"OOF artifact changed: {name}")
            paths.append(path)
        settings = self.read(run / "config.json")
        if settings["input_and_code_sha256"] != c["hashes"] or settings["versions"] != c["versions"]:
            raise ValueError("OOF source inputs/code/environment differ. Inspect; use --no-cache to refit rather than bypass checks.")
        p = pd.read_csv(run / "validation_probabilities.csv", dtype={"id": "string"}, float_precision="round_trip")
        keys = ["repeat", "split_seed", "fold", "row_index"]
        if len(p) != 10 * len(c["x"]) or p.duplicated(keys).any():
            raise ValueError("OOF row count/keys invalid.")
        for key in keys:
            values = p[key].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not (values == np.floor(values)).all():
                raise ValueError("OOF keys must be finite integers.")
        indexed = p.set_index(keys, verify_integrity=True)
        expected_keys = []
        scores = pd.read_csv(run / "fold_scores.csv").set_index(["repeat", "fold"], verify_integrity=True)
        if len(scores) != 50:
            raise ValueError("OOF fold-score table incomplete.")
        for split in c["splits"]:
            indices = split["valid_indices"]
            fold_keys = [(split["repeat"], split["split_seed"], split["fold"], i) for i in indices]
            expected_keys.extend(fold_keys)
            f = indexed.loc[fold_keys]
            if f["id"].tolist() != c["ids"].iloc[indices].astype(str).tolist() or not np.array_equal(f["target"], c["y"].iloc[indices]):
                raise ValueError("OOF IDs/labels differ from verified validation rows.")
            blend = soft_vote({"l1": f.l1, "xgboost": f.xgboost}, ["l1", "xgboost"], [1, 1])
            if not np.allclose(f.ensemble, blend, rtol=0, atol=1e-12):
                raise ValueError("Cached ensemble is not the original 50/50 blend.")
            key = (split["repeat"], split["fold"])
            for name, values in (("l1", f.l1), ("xgboost", f.xgboost), ("ensemble", blend)):
                auc = roc_auc_score(c["y"].iloc[indices], values)
                if not np.isclose(auc, scores.loc[key, name], rtol=0, atol=1e-12):
                    raise ValueError("Cached probabilities do not reproduce source AUC.")
            if not np.isclose(scores.loc[key, "xgboost"], c["expected"].loc[key], rtol=0, atol=1e-12):
                raise ValueError("OOF XGBoost differs from frozen control.")
        if set(indexed.index) != set(expected_keys):
            raise ValueError("OOF split assignments differ.")
        return indexed, paths

    def _prepare(self, config_path, output=None, no_cache=False):
        from src.experiments.soft_voting_challenge import SoftVotingChallenge
        from src.modeling import registered_models
        source = SoftVotingChallenge()
        # Keep the old runner and its source hashes unchanged for historical OOF reuse.
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        path = FrozenXGBoostSubmission._path(config_path)
        cfg = self.validate(self.read(path))
        c = source._prepare(cfg["frozen_dir"], cfg["l1_config"], output or cfg["output_dir"])
        self.helper = source.helper
        cache, extra = None, []
        if cfg["reuse_cv_run"] and not no_cache:
            cache, extra = self._cache(c, self.helper._path(cfg["reuse_cv_run"]))
        versions = dict(c["versions"])
        if any(s["type"] == "lgbm_classifier" for s in cfg["models"].values()):
            try:
                versions["lightgbm"] = version("lightgbm")
                from lightgbm import LGBMClassifier
            except Exception as exc:
                raise ImportError(f"LightGBM environment check failed: {exc}. Install: python -m pip install -r requirements-registry.txt") from exc
        extra += [path, Path(__file__), Path(inspect.getfile(ModelRegistry)), Path(inspect.getfile(registered_models))]
        hashes = dict(c["hashes"])
        hashes.update({str(p.resolve()): self.helper._sha(p) for p in extra})
        c.update(cfg=cfg, config_path=path, hashes=hashes, versions=versions, cache=cache, no_cache=no_cache)
        for spec in cfg["models"].values():
            ModelRegistry.create(spec["type"], context=c, parameters=spec.get("parameters"))
        return c

    def _unchanged(self, c):
        for path, digest in c["hashes"].items():
            if self.helper._sha(path) != digest:
                raise RuntimeError(f"Input or code changed: {path}")

    def _fit_predict(self, c, alias, train, y, valid):
        spec = c["cfg"]["models"][alias]
        model = ModelRegistry.create(spec["type"], context=c, parameters=spec.get("parameters"))
        model.fit(train, y)
        p = model.positive_probability(valid)
        return model, p

    @staticmethod
    def _cost(cfg, names):
        return sum(2 if cfg["models"][n]["type"] == "xgb_topk" else 1 for n in names)

    @staticmethod
    def summarize(folds, cfg):
        names = [*cfg["models"], *cfg["ensembles"]]
        repeats = folds.groupby(["repeat", "split_seed"], as_index=False)[names].mean()
        rows = []
        for name in names:
            d = repeats[name] - repeats[cfg["baseline"]]
            candidate = name in cfg["ensembles"] and name != cfg["baseline"]
            rows.append(dict(candidate=name, mean_auc=float(repeats[name].mean()), repeat_sd=float(repeats[name].std(ddof=1)),
                difference_vs_baseline=float(d.mean()), wins=int((d > 1e-12).sum()), losses=int((d < -1e-12).sum()),
                ties=int((abs(d) <= 1e-12).sum()), promotable=bool(candidate and d.mean() > 1e-12 and (d > 1e-12).sum() >= 6)))
        table = pd.DataFrame(rows)
        eligible = table.loc[table.promotable]
        selected = None
        if len(eligible):
            best = eligible.mean_auc.max()
            tied = eligible.loc[abs(eligible.mean_auc - best) <= 1e-12, "candidate"].tolist()
            selected = min(tied, key=lambda n: (len(cfg["ensembles"][n]["members"]), n))
        decision = dict(selected=selected, baseline=cfg["baseline"],
            rule="Positive mean paired AUC difference vs existing L1+XGBoost and >=6/10 repeat wins; not a significance test.",
            interpretation="Adaptive development comparison on reused splits; no independent final score.")
        return repeats, table, decision

    def cv(self, config_path, output=None, no_cache=False, dry_run=False):
        c = self._prepare(config_path, output, no_cache)
        h, cfg = self.helper, c["cfg"]
        fitted = [n for n in cfg["models"] if c["cache"] is None or n not in ("l1", "xgboost")]
        planned = len(c["splits"]) * self._cost(cfg, fitted)
        if dry_run:
            return dict(status="preflight_passed", new_fits=planned, cached_models=[] if c["cache"] is None else ["l1", "xgboost"], test_opened=False)
        out = c["output"]; out.mkdir(parents=True, exist_ok=False)
        h._write(out / "settings.json", dict(config=cfg, config_path=str(c["config_path"]), no_cache=no_cache,
            input_and_code_sha256=c["hashes"], versions=c["versions"]))
        manifest = dict(status="running", completed_folds=0, planned_fits=planned, test_opened=False, kaggle_upload_performed=False)
        rows, predictions, diagnostics, warning_records = [], [], [], []
        try:
            for split in c["splits"]:
                self._unchanged(c)
                tr, va = split["train_indices"], split["valid_indices"]
                key = {k: split[k] for k in ("repeat", "split_seed", "fold")}
                probabilities = {}
                if c["cache"] is not None:
                    f = c["cache"].loc[[(split["repeat"], split["split_seed"], split["fold"], i) for i in va]]
                    probabilities.update(l1=f.l1.to_numpy(), xgboost=f.xgboost.to_numpy())
                caught = []
                try:
                    with warnings.catch_warnings(record=True) as caught, (out / "fit.log").open("a", encoding="utf-8") as log:
                        warnings.simplefilter("always")
                        with redirect_stdout(log), redirect_stderr(log):
                            for alias in fitted:
                                model, p = self._fit_predict(c, alias, c["x"].iloc[tr], c["y"].iloc[tr], c["x"].iloc[va])
                                probabilities[alias] = p
                                diagnostics.append(dict(key, model=alias, unique_probabilities=int(np.unique(p).size), **model.diagnostics()))
                finally:
                    warning_records.extend(dict(key, category=w.category.__name__, message=str(w.message)) for w in caught)
                    h._write(out / "warnings.json", warning_records)
                if caught:
                    raise RuntimeError("Python warnings require review; inspect warnings.json and fit.log.")
                values = self.assemble(probabilities, cfg["ensembles"])
                auc = {name: float(roc_auc_score(c["y"].iloc[va], p)) for name, p in values.items()}
                if not np.isclose(auc["xgboost"], c["expected"].loc[(split["repeat"], split["fold"])], rtol=0, atol=1e-12):
                    raise ValueError("XGBoost control does not reproduce; do not relax tolerance.")
                rows.append(dict(key, **auc))
                for j, index in enumerate(va):
                    predictions.append(dict(key, row_index=int(index), id=str(c["ids"].iloc[index]), target=int(c["y"].iloc[index]),
                                            **{name: float(p[j]) for name, p in values.items()}))
                pd.DataFrame(rows).to_csv(out / "fold_scores.csv", index=False)
                pd.DataFrame(predictions).to_csv(out / "validation_probabilities.csv", index=False)
                h._write(out / "diagnostics.json", diagnostics)
                manifest["completed_folds"] = len(rows); h._write(out / "run_manifest.json", manifest)
                print(f"[{len(rows)}/{len(c['splits'])}] {auc}", flush=True)
            self._unchanged(c)
            repeats, table, decision = self.summarize(pd.DataFrame(rows), cfg)
            repeats.to_csv(out / "repeat_scores.csv", index=False); table.to_csv(out / "comparison.csv", index=False)
            h._write(out / "decision.json", decision)
            manifest.update(status="completed", fits_performed=planned, control_auc_reproduced=True)
            manifest["artifact_sha256"] = {p.name: h._sha(p) for p in out.iterdir() if p.is_file() and p.name != "run_manifest.json"}
            return dict(decision=decision, comparison=table.to_dict("records"))
        except BaseException as exc:
            manifest.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
            h._write(out / "error.json", dict(error=manifest["error"], traceback=traceback.format_exc())); raise
        finally:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(); h._write(out / "run_manifest.json", manifest)

    def predict(self, run_path, output, candidate="auto", test=None, template=None, dry_run=False):
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        h = FrozenXGBoostSubmission(); run = h._path(run_path)
        manifest = self.read(run / "run_manifest.json")
        if manifest.get("status") != "completed" or manifest.get("completed_folds") != 50 or not manifest.get("control_auc_reproduced"):
            raise ValueError("Need a completed, control-reproduced 50-fold registry study.")
        for name, digest in manifest["artifact_sha256"].items():
            path = (run / name).resolve()
            if not path.is_relative_to(run) or h._sha(path) != digest:
                raise ValueError("CV artifacts changed.")
        settings, decision = self.read(run / "settings.json"), self.read(run / "decision.json")
        name = decision["selected"] if candidate == "auto" else candidate
        table = pd.read_csv(run / "comparison.csv")
        row = table.loc[table.candidate == name]
        if name is None or len(row) != 1 or not bool(row.iloc[0].promotable):
            raise ValueError("No promoted candidate; retain existing 0.91514 submission. Do not bypass the rule.")
        c = self._prepare(settings["config_path"], output, settings["no_cache"])
        h = self.helper
        if c["hashes"] != settings["input_and_code_sha256"] or c["versions"] != settings["versions"] or c["cfg"] != settings["config"]:
            raise ValueError("Inputs/code/settings/environment differ from evaluated study.")
        spec = c["cfg"]["ensembles"][name]
        test = h._path(test or c["candidate"]["default_test_path"])
        template = h._path(template or c["candidate"]["default_template_path"])
        xt, ids, target = h._test_inputs(test, template, c["features"], c["base"].id_column,
                                        c["base"].target_column, c["candidate"]["expected_test_rows"])
        if dry_run:
            return dict(status="prediction_preflight_passed", candidate=name, rows=len(xt), fits_performed=0)
        out = c["output"]; out.mkdir(parents=True, exist_ok=False); (out / "models").mkdir()
        record = dict(status="running", candidate=name, source_cv_run=str(run), versions=c["versions"],
            kaggle_upload_performed=False, input_sha256={str(p): h._sha(p) for p in (test, template)})
        caught = []; probabilities = {}; details = {}
        try:
            with warnings.catch_warnings(record=True) as caught, (out / "fit.log").open("w", encoding="utf-8") as log:
                warnings.simplefilter("always")
                with redirect_stdout(log), redirect_stderr(log):
                    for alias in spec["members"]:
                        model, before = self._fit_predict(c, alias, c["x"], c["y"], xt)
                        path = out / "models" / f"{alias}.joblib"; model.save(path)
                        loaded = RegisteredModel.load(path); after = loaded.positive_probability(xt)
                        if not np.allclose(before, after, rtol=0, atol=1e-12):
                            raise ValueError("Reloaded model predictions differ.")
                        probabilities[alias] = after; details[alias] = loaded.diagnostics()
            if caught:
                raise RuntimeError("Python warnings require review; no submission written.")
            self._unchanged(c)
            for path, digest in record["input_sha256"].items():
                if h._sha(path) != digest:
                    raise ValueError("Prediction input changed.")
            scores = soft_vote(probabilities, spec["members"], spec["weights"])
            frame = h._align(ids, scores, target, c["base"].id_column, c["base"].target_column)
            frame.to_csv(out / "submission.csv", index=False)
            reread = h._read_csv(out / "submission.csv", c["base"].id_column)
            if not reread[c["base"].id_column].equals(target[c["base"].id_column]) or not np.allclose(
                    reread[c["base"].target_column], frame[c["base"].target_column], rtol=0, atol=1e-12):
                raise ValueError("CSV round-trip/ID alignment failed.")
            h._write(out / "ensemble.json", dict(candidate=name, specification=spec, models=c["cfg"]["models"],
                feature_order=c["features"], model_diagnostics=details,
                l1_configuration=c["linear"].to_dict(), xgboost_parameters=c["candidate"]["estimator_parameters"]))
            record.update(status="ready_for_manual_submission", rows=len(frame), fits_performed=self._cost(c["cfg"], spec["members"]),
                saved_and_reloaded_models_verified=True, submission_sha256=h._sha(out / "submission.csv"))
            record["artifact_sha256"] = {str(p.relative_to(out)): h._sha(p) for p in out.rglob("*") if p.is_file()}
            return dict(status=record["status"], candidate=name, submission=str(out / "submission.csv"), rows=len(frame))
        except BaseException as exc:
            record.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
            h._write(out / "error.json", dict(error=record["error"], traceback=traceback.format_exc())); raise
        finally:
            h._write(out / "warnings.json", [dict(category=w.category.__name__, message=str(w.message)) for w in caught])
            record["finished_utc"] = datetime.now(timezone.utc).isoformat(); h._write(out / "prediction_manifest.json", record)

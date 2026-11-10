"""Full-data Trial25 refit, model persistence and manual-submission export."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import inspect
from pathlib import Path
import traceback
import warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.modeling.elasticnet_trial_model import ElasticnetTrialModel
from src.modeling.model_registry import RegisteredModel
from src.submission.elasticnet_trial_snapshot import ElasticnetTrialSnapshot


class ElasticnetTrialSubmission:
    candidate_name = "elasticnet_optuna_trial_25"

    def __init__(self, snapshot=None, model_factory=None):
        self.snapshot = snapshot or ElasticnetTrialSnapshot()
        self.model_factory = model_factory or self.create_model

    @staticmethod
    def create_model(alias, c, parameters):
        from src.modeling.registered_models import L1Model, XGBTopKModel, LightGBMModel
        if alias == "l1":
            return L1Model(context=c["context"])
        if alias == "xgboost":
            return XGBTopKModel(context=c["context"])
        if alias == "lightgbm":
            return LightGBMModel(parameters=c["lightgbm_parameters"])
        if alias == "elasticnet":
            return ElasticnetTrialModel({k: parameters[k] for k in ("C", "l1_ratio")})
        raise ValueError("Unknown branch: " + alias)

    def prepare(self, run, config, output, test=None, template=None):
        saved = self.snapshot.load(run, config, output)
        c, h = saved["context"], self.snapshot.helper
        ctx = c["context"]
        features = list(ctx["x"].columns)
        test_path = h._path(test or ctx["candidate"]["default_test_path"])
        template_path = h._path(template or ctx["candidate"]["default_template_path"])
        if any(p.is_relative_to(saved["output"]) for p in (test_path, template_path)):
            raise ValueError("Output overlaps prediction inputs")
        # Hash BEFORE reading: any edit during the read/refit is detected before publication.
        inputs = {str(p): h._sha(p) for p in (test_path, template_path)}
        xt, ids, target = h._test_inputs(test_path, template_path, features,
                        ctx["base"].id_column, ctx["base"].target_column,
                        ctx["candidate"]["expected_test_rows"])
        self.snapshot.verify_hashes(inputs)
        code = [Path(inspect.getfile(cls)).resolve() for cls in
                (ElasticnetTrialSubmission, ElasticnetTrialSnapshot, ElasticnetTrialModel)]
        cli = h._path("scripts/predict_elasticnet_trial_25.py")
        if cli.is_file():
            code.append(cli)
        saved["hashes"].update({str(p): h._sha(p) for p in code})
        saved.update(test=xt, test_ids=ids, template=target, features=features, input_hashes=inputs)
        return saved

    @staticmethod
    def blend(probabilities, alpha):
        original = (probabilities["l1"] + probabilities["xgboost"] + probabilities["lightgbm"]) / 3
        return original, (1 - alpha) * original + alpha * probabilities["elasticnet"]

    def predict(self, run="outputs/results/elasticnet_optuna_v1_run_01",
                config="configs/elasticnet_optuna_v1.json",
                output="outputs/submissions/elasticnet_optuna_trial_25_run_01",
                test=None, template=None, dry_run=False):
        saved = self.prepare(run, config, output, test, template)
        c, out, h = saved["context"], saved["output"], self.snapshot.helper
        if dry_run:
            return dict(status="prediction_preflight_passed", candidate=self.candidate_name,
                        selected_trial=25, train_rows=len(c["context"]["x"]), rows=len(saved["test"]),
                        planned_estimator_fits=5, fits_performed=0, test_opened=True,
                        output_created=False, kaggle_upload_performed=False, output=str(out))
        out.mkdir(parents=True, exist_ok=False)
        (out / "models").mkdir()
        manifest = dict(status="running", candidate=self.candidate_name, selected_trial=25,
                        source_study=str(saved["run"]), started_utc=datetime.now(timezone.utc).isoformat(),
                        planned_estimator_fits=5, attempted_estimator_fits=0, fits_performed=0,
                        test_opened=True, kaggle_upload_performed=False, automatic_promotion=False,
                        train_rows=len(c["context"]["x"]), parameters=saved["parameters"],
                        versions=dict(c.get("versions", {}), **saved["settings"]["versions"]),
                        source_input_and_code_sha256=saved["hashes"],
                        prediction_input_sha256=saved["input_hashes"],
                        interpretation="Full-data refit/export of adaptive-development Trial25; not an independent evaluation.")
        probabilities, details, warning_rows = {}, {}, []
        final_csv, pending_csv = out / "submission.csv", out / "submission.pending.csv"
        h._write(out / "prediction_manifest.json", manifest)
        try:
            h._write(out / "source_decision.json", saved["decision"])
            h._write(out / "source_best_trial.json", saved["best"])
            with threadpool_limits(limits=1):
                for alias in ("l1", "xgboost", "lightgbm", "elasticnet"):
                    self.snapshot.verify_hashes(saved["hashes"])
                    cost = 2 if alias == "xgboost" else 1
                    manifest["attempted_estimator_fits"] += cost
                    h._write(out / "prediction_manifest.json", manifest)
                    caught = []
                    try:
                        with warnings.catch_warnings(record=True) as caught, (out / "fit.log").open("a", encoding="utf-8") as log:
                            warnings.simplefilter("always")
                            with redirect_stdout(log), redirect_stderr(log):
                                model = self.model_factory(alias, c, saved["parameters"])
                                model.fit(c["context"]["x"], c["context"]["y"])
                                manifest["fits_performed"] += cost
                                before = self.snapshot.probability(model.positive_probability(saved["test"]), len(saved["test"]))
                                path = out / "models" / (alias + ".joblib")
                                model.save(path)
                                loaded = RegisteredModel.load(path)
                                after = self.snapshot.probability(loaded.positive_probability(saved["test"]), len(saved["test"]))
                                if not np.allclose(before, after, rtol=0, atol=1e-12):
                                    raise ValueError("Reloaded model predictions differ: " + alias)
                                probabilities[alias] = after
                                details[alias] = loaded.diagnostics()
                    finally:
                        warning_rows.extend(dict(model=alias, category=w.category.__name__, message=str(w.message)) for w in caught)
                        h._write(out / "warnings.json", warning_rows)
                    if caught:
                        raise RuntimeError("Python warnings require review; no final submission written")
            original, blended = self.blend(probabilities, saved["parameters"]["alpha"])
            blended = self.snapshot.probability(blended, len(saved["test"]))
            id_column = c["context"]["base"].id_column
            target_column = c["context"]["base"].target_column
            audit = pd.DataFrame({id_column: saved["test_ids"], **probabilities,
                                  "original_three_model": original, "ensemble": blended})
            audit.to_csv(out / "component_predictions.csv", index=False)
            h._write(out / "ensemble.json", dict(candidate=self.candidate_name,
                     formula="(1-alpha)*(p_l1+p_xgboost+p_lightgbm)/3+alpha*p_elasticnet",
                     alpha=saved["parameters"]["alpha"], feature_order=saved["features"],
                     model_diagnostics=details, elasticnet_parameters=saved["parameters"],
                     l1_configuration=c["context"]["linear"].to_dict(),
                     xgboost_parameters=c["context"]["candidate"]["estimator_parameters"],
                     xgboost_k=c["context"]["candidate"]["k"],
                     lightgbm_parameters=c["lightgbm_parameters"],
                     model_files={alias: "models/" + alias + ".joblib" for alias in probabilities}))
            frame = h._align(saved["test_ids"], blended, saved["template"], id_column, target_column)
            frame.to_csv(pending_csv, index=False)
            reread = h._read_csv(pending_csv, id_column)
            if (list(reread.columns) != [id_column, target_column] or len(reread) != len(saved["test"])
                    or not reread[id_column].equals(saved["template"][id_column])
                    or not np.allclose(reread[target_column], frame[target_column], rtol=0, atol=1e-12)):
                raise ValueError("Submission CSV round-trip/ID alignment failed")
            self.snapshot.probability(reread[target_column], len(frame))
            self.snapshot.verify_hashes(saved["hashes"])
            self.snapshot.verify_hashes(saved["input_hashes"])
            if manifest["fits_performed"] != 5:
                raise ValueError("Full-data estimator fit budget changed")
            pending_csv.replace(final_csv)
            manifest.update(rows=len(frame), saved_and_reloaded_models_verified=True,
                            submission_sha256=h._sha(final_csv),
                            prediction_min=float(blended.min()), prediction_max=float(blended.max()))
            manifest["artifact_sha256"] = {str(p.relative_to(out)): h._sha(p) for p in sorted(out.rglob("*"))
                                           if p.is_file() and p.name != "prediction_manifest.json"}
            manifest["status"] = "ready_for_manual_submission"
            return dict(status=manifest["status"], candidate=self.candidate_name,
                        submission=str(final_csv), rows=len(frame), fits_performed=5)
        except BaseException as exc:
            # Never leave a final-named CSV behind after failed validation.
            final_csv.unlink(missing_ok=True)
            pending_csv.unlink(missing_ok=True)
            manifest.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
            h._write(out / "error.json", dict(error=manifest["error"], traceback=traceback.format_exc()))
            raise
        finally:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            h._write(out / "prediction_manifest.json", manifest)

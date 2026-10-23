"""Run E5b on saved splits; reuse the project's ValidationResult aggregation."""
from __future__ import annotations
import hashlib
import inspect
import json
import logging
import math
import platform
import subprocess
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from src.data.labelled_data_loader import LabelledDataLoader
from src.modeling.xgboost_pipeline_factory import XGBoostPipelineFactory
from src.validation.saved_split_loader import SavedSplitLoader
from src.validation.paired_comparison import PairedComparison
from src.validation.validation_result import ValidationResult


class XGBoostExperimentRunner:
    def __init__(self, factory=None):
        self.factory = factory if factory is not None else XGBoostPipelineFactory()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _sha256(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1048576), b""):
                h.update(block)
        return h.hexdigest()

    @classmethod
    def _jsonable(cls, obj):
        if isinstance(obj, dict):
            return {str(k): cls._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [cls._jsonable(v) for v in obj]
        if isinstance(obj, np.generic):
            return cls._jsonable(obj.item())
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, float) and not math.isfinite(obj):
            return "NaN" if math.isnan(obj) else "Infinity" if obj > 0 else "-Infinity"
        return obj

    @classmethod
    def _save_json(cls, path, value):
        Path(path).write_text(json.dumps(cls._jsonable(value), ensure_ascii=False,
                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @staticmethod
    def _git_info(root):
        try:
            rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=5)
            state = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5)
            return dict(revision=rev.stdout.strip() if rev.returncode == 0 else None,
                        dirty=bool(state.stdout.strip()) if state.returncode == 0 else None)
        except (OSError, subprocess.SubprocessError):
            return dict(revision=None, dirty=None)

    def _prepare(self, config, confirm_reference_alignment):
        if not confirm_reference_alignment:
            raise ValueError("Confirm unchanged historical data, row order and metric with --confirm-reference-alignment.")
        if config.output_dir.exists():
            raise FileExistsError(f"Output already exists: {config.output_dir}. Choose a new directory.")
        version = self.factory.check_environment()
        x, y, ids, features = LabelledDataLoader().load(config)
        loader = SavedSplitLoader()
        splits = loader.load(config.split_indices_path, y, features)
        sparse_splits = loader.load(config.sparse_split_indices_path, y, features)
        if loader.signature(splits) != loader.signature(sparse_splits):
            raise ValueError("The two E4 references used different saved split assignments.")
        references = PairedComparison().load_references(config.reference_scores_path)
        return x, y, ids, features, splits, references, version

    def _fit_fold(self, config, x, y, ids, split, experiment_id, depth, trees):
        key = {k: split[k] for k in ("repeat", "split_seed", "fold")}
        key.update(experiment_id=experiment_id, max_depth=depth, n_estimators=trees)
        tr, va = split["train_indices"], split["valid_indices"]
        record = dict(key, n_train=len(tr), n_validation=len(va), valid=False,
                      auc=None, error="", fit_seconds=None, rounds_fitted=None)
        predictions, booster_config, parameters, caught = [], None, None, []
        start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline = self.factory.create(config, depth, trees)
                classifier = pipeline.named_steps["classifier"]
                parameters = classifier.get_params(deep=False)
                pipeline.fit(x.iloc[tr], y.iloc[tr])
                record["fit_seconds"] = time.perf_counter() - start
                if not np.array_equal(pipeline.classes_, [0, 1]):
                    raise ValueError("Unexpected class order.")
                proba = np.asarray(pipeline.predict_proba(x.iloc[va]))
                if proba.shape != (len(va), 2) or not np.isfinite(proba).all():
                    raise ValueError("Invalid or non-finite probability shape.")
                if not ((proba >= 0) & (proba <= 1)).all() or not np.allclose(proba.sum(axis=1), 1., atol=1e-6):
                    raise ValueError("Invalid probability range or row sums.")
                scores = proba[:, 1]
                booster = classifier.get_booster()
                rounds = int(booster.num_boosted_rounds())
                if rounds != trees:
                    raise ValueError("Fitted rounds differ from fixed n_estimators; no early stopping permitted.")
                booster_config = json.loads(booster.save_config())
                record.update(valid=True, auc=float(roc_auc_score(y.iloc[va], scores)),
                              rounds_fitted=rounds, unique_scores=int(np.unique(scores).size),
                              score_min=float(scores.min()), score_max=float(scores.max()),
                              score_std=float(scores.std(ddof=0)),
                              features_used=len(booster.get_score(importance_type="weight")))
                predictions = [dict(key, row_index=int(i), id=str(ids.iloc[i]),
                                    target=int(y.iloc[i]), probability_target_1=float(score))
                               for i, score in zip(va, scores)]
        except Exception as exc:
            record.update(valid=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if record["fit_seconds"] is None:
            record["fit_seconds"] = time.perf_counter()-start
        record["total_seconds"] = time.perf_counter()-start
        record["warnings_json"] = json.dumps([dict(category=w.category.__name__, message=str(w.message)) for w in caught])
        record["warning_count"] = len(caught)
        record["review_required"] = bool(caught or not record["valid"])
        for p in predictions:
            p["fold_valid"] = record["valid"]
        return record, predictions, booster_config, parameters

    def _summarize(self, config, records, references):
        folds, rows, repeats = pd.DataFrame(records), [], []
        for name, depth, trees in config.candidates():
            f = folds.loc[folds["experiment_id"] == name]
            if f.empty:
                continue
            complete = len(f) == 50 and bool(f["valid"].all())
            review = bool(f["review_required"].any())
            row = dict(experiment_id=name, max_depth=depth, n_estimators=trees,
                       completed_folds=len(f), valid_folds=int(f["valid"].sum()),
                       failed_folds=int((~f["valid"]).sum()),
                       warning_folds=int((f["warning_count"] > 0).sum()),
                       warnings_total=int(f["warning_count"].sum()), complete_valid=complete,
                       review_required=review, eligible=bool(complete and not review),
                       total_fit_seconds=float(f["fit_seconds"].sum()))
            if complete:
                score_frame = f.rename(columns={"n_train": "train_rows", "n_validation": "valid_rows", "auc": "valid_auc"})
                result = ValidationResult(score_frame)
                result.save(config.output_dir / name)
                row.update(result.summary())
                r, paired = PairedComparison().compare(result.repeat_scores(), references)
                row.update(paired)
                r["experiment_id"], r["max_depth"], r["n_estimators"] = name, depth, trees
                r["review_required"] = review
                repeats.extend(r.to_dict("records"))
            rows.append(row)
        comparison = pd.DataFrame(rows)
        comparison.to_csv(config.output_dir / "e5b_comparison.csv", index=False)
        repeat_table = pd.DataFrame(repeats) if repeats else pd.DataFrame(columns=["experiment_id", "repeat", "split_seed", "mean_auc"])
        repeat_table.to_csv(config.output_dir / "e5b_paired_repeat_scores.csv", index=False)
        return comparison

    @staticmethod
    def _choose(comparison, references):
        eligible = comparison.loc[comparison["eligible"]]
        if eligible.empty:
            return dict(status="no_eligible_candidate", e5b_leader=None, promote_over_e4=False)
        best = eligible["mean_auc"].max()
        tied = eligible.loc[abs(eligible["mean_auc"] - best) <= PairedComparison.TOLERANCE]
        winner = tied.sort_values(["max_depth", "n_estimators"]).iloc[0]
        difference = float(winner["mean_auc"] - references["e4_leader_auc"].mean())
        return dict(status="development_selection_only", e5b_leader=winner["experiment_id"],
                    mean_auc=float(winner["mean_auc"]), mean_difference_vs_e4_leader=difference,
                    promote_over_e4=bool(difference > PairedComparison.TOLERANCE),
                    note="A provisional mean ordering, not statistical proof or independent final evaluation.")

    def run(self, config, dry_run=False, confirm_reference_alignment=False):
        x, y, ids, features, splits, references, version = self._prepare(config, confirm_reference_alignment)
        output = config.output_dir
        output.mkdir(parents=True, exist_ok=False)
        self._save_json(output / "config.json", config.to_dict())
        self._save_json(output / "verified_split_indices.json", dict(
            index_type="zero_based_row_position", n_rows=len(y), n_splits=5,
            split_seeds=list(range(10)), feature_names=features, splits=splits))
        references.to_csv(output / "reference_repeat_scores.csv", index=False)
        source_classes = (type(config), LabelledDataLoader, XGBoostPipelineFactory,
                          SavedSplitLoader, PairedComparison, ValidationResult, XGBoostExperimentRunner)
        source_hashes = {str(Path(inspect.getfile(c)).resolve()): self._sha256(inspect.getfile(c)) for c in source_classes}
        cli_source = config.project_root() / "scripts/run_e5b.py"
        if cli_source.exists():
            source_hashes[str(cli_source)] = self._sha256(cli_source)
        manifest = dict(status="validated", dry_run=bool(dry_run), started_utc=datetime.now(timezone.utc).isoformat(),
                        planned_candidates=len(list(config.candidates())), planned_fits=50*len(list(config.candidates())),
                        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                      scikit_learn=sklearn.__version__, xgboost=version),
                        fixed_parameters=self.factory.parameters(config, config.max_depth_values[0], config.n_estimators_values[0]),
                        candidates=[dict(experiment_id=n,max_depth=d,n_estimators=t) for n,d,t in config.candidates()],
                        feature_order=features, split_signature=SavedSplitLoader.signature(splits),
                        input_sha256={name:self._sha256(getattr(config,name)) for name in
                                      ("train_path","split_indices_path","sparse_split_indices_path","reference_scores_path")},
                        source_sha256=source_hashes, git=self._git_info(config.project_root()),
                        metric="mean_fold_auc_per_repeat", aggregation_class="src.validation.validation_result.ValidationResult",
                        score_method="predict_proba_target_1", scaling="none", feature_selection=None, early_stopping=False,
                        reference_alignment="Saved assignments verified; historical data contents/order and metric user-attested.",
                        warning_policy="Every captured Python warning requires review and prevents automatic eligibility.",
                        json_metadata_note="Non-finite estimator parameter values are encoded as NaN/Infinity strings.")
        manifest_path = output / "run_manifest.json"
        self._save_json(manifest_path, manifest)
        if dry_run:
            self.logger.info("E5b preflight passed. No fits. Use a new output directory for the full run.")
            return 0
        all_records = []
        manifest["status"] = "running"
        self._save_json(manifest_path, manifest)
        try:
            for name, depth, trees in config.candidates():
                dest = output / name
                dest.mkdir()
                self._save_json(dest / "config.json", dict(experiment_id=name,
                    estimator_parameters=self.factory.parameters(config,depth,trees), scaling="none", early_stopping=False))
                local, predictions = [], []
                with (dest / "booster_configs.jsonl").open("w", encoding="utf-8") as booster_file:
                    for split in splits:
                        rec, pred, booster, parameters = self._fit_fold(config,x,y,ids,split,name,depth,trees)
                        all_records.append(rec)
                        local.append(rec)
                        predictions.extend(pred)
                        pd.DataFrame(local).to_csv(dest / "fold_diagnostics.csv", index=False)
                        columns = ["experiment_id","repeat","split_seed","fold","max_depth","n_estimators",
                                   "row_index","id","target","probability_target_1","fold_valid"]
                        pd.DataFrame(predictions,columns=columns).to_csv(dest / "validation_probabilities.csv", index=False)
                        pd.DataFrame(all_records).to_csv(output / "e5b_fold_diagnostics.csv", index=False)
                        if booster is not None:
                            booster_file.write(json.dumps(self._jsonable(dict(repeat=split["repeat"],fold=split["fold"],config=booster)),allow_nan=False)+"\n")
                            booster_file.flush()
                        if parameters is not None and split["repeat"] == 0 and split["fold"] == 0:
                            self._save_json(dest / "estimator_parameters.json", parameters)
                        self.logger.info("[%d/%d] %s repeat=%d fold=%d AUC=%s valid=%s warnings=%d",
                            len(all_records),manifest["planned_fits"],name,split["repeat"],split["fold"],
                            "NA" if rec["auc"] is None else f"{rec['auc']:.6f}",rec["valid"],rec["warning_count"])
                self._summarize(config,all_records,references)
            comparison = self._summarize(config,all_records,references)
            self._save_json(output / "e5b_decision.json", self._choose(comparison,references))
            manifest.update(status="completed", completed_fits=len(all_records), eligible_candidates=int(comparison["eligible"].sum()))
            self.logger.info("E5b finished. Review %s", output / "e5b_comparison.csv")
            return 0 if bool(comparison["eligible"].all()) else 2
        except BaseException:
            manifest.update(status="interrupted_or_failed", completed_fits=len(all_records))
            raise
        finally:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            self._save_json(manifest_path,manifest)

"""Freeze the selected fold-local procedure, then refit/export without uploading."""
import csv
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import traceback
import warnings

import numpy as np
import pandas as pd
import sklearn


class FrozenXGBoostSubmission:
    def __init__(self,model_type=None):
        self.model_type=model_type

    @staticmethod
    def _root():
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _path(cls,path):
        return (cls._root()/Path(path).expanduser()).resolve()

    @staticmethod
    def _sha(path):
        h=hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda:stream.read(1048576),b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _write(path,value):
        Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")

    def _versions(self):
        if self.model_type is None:
            import xgboost
            self.model_type=xgboost.XGBClassifier
            xgb_version=xgboost.__version__
        else:
            xgb_version=getattr(self.model_type,"test_version","INJECTED_MODEL")
        return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                    sklearn=sklearn.__version__,xgboost=xgb_version)

    @staticmethod
    def _verify_completed_extension(source,k):
        manifest=json.loads((source/"run_manifest.json").read_text())
        if manifest.get("status")!="completed" or manifest.get("completed_fits")!=manifest.get("planned_fits"):
            raise ValueError("Feature-selection study is not completed.")
        if manifest.get("control_auc_reproduced") is not True:
            raise ValueError("Control AUC reproduction was not confirmed.")
        checks=pd.read_csv(source/"control_auc_reproduction.csv")
        if len(checks)!=50 or not checks["matches"].eq(True).all() or checks.duplicated(["repeat","fold"]).any():
            raise ValueError("Expected 50 successful unique control checks.")
        comparison=pd.read_csv(source/"feature_selection_comparison.csv")
        rows=comparison.loc[comparison["k"]==k]
        if len(rows)!=1 or not bool(rows.iloc[0]["eligible"]) or int(rows.iloc[0]["valid_folds"])!=50:
            raise ValueError("Requested k is not one eligible 50-fold candidate.")
        for path,expected in manifest["input_and_source_sha256"].items():
            if FrozenXGBoostSubmission._sha(path)!=expected:
                raise ValueError(f"Historical input/source changed: {path}.")
        return manifest,rows.iloc[0]

    def freeze(self,source_run,k,frozen_dir,confirmed=False):
        if not confirmed:
            raise ValueError("Pass --confirm-reference-alignment after checking historical alignment.")
        source=self._path(source_run);dest=self._path(frozen_dir)
        if dest.exists():raise FileExistsError("Frozen directory already exists; never overwrite a candidate version.")
        manifest,row=self._verify_completed_extension(source,k)
        from src.config.xgboost_feature_selection_config import XGBoostFeatureSelectionConfig
        from src.experiments.xgboost_feature_selection_runner import XGBoostFeatureSelectionRunner
        source_config=XGBoostFeatureSelectionConfig.from_json(source/"config.json")
        # Reuse source validation without fitting; give it an unused output destination.
        context=XGBoostFeatureSelectionRunner()._prepare(replace(source_config,output_dir=dest),True)
        if not 1<=k<len(context["features"]):raise ValueError("Expected a selected-feature candidate, not the all-feature control.")
        candidate_id=str(row["experiment_id"])
        if Path(candidate_id).name!=candidate_id:raise ValueError("Unsafe candidate ID.")
        evidence_files=["config.json","run_manifest.json","feature_selection_comparison.csv",
            "feature_selection_paired_repeat_scores.csv","control_auc_reproduction.csv",
            "source_trial_parameters.json","verified_split_indices.json","reference_repeat_scores.csv",
            "selected_features.json","selection_frequencies.csv"]
        for name in evidence_files:
            if not (source/name).is_file():raise FileNotFoundError(source/name)
        candidate=source/candidate_id
        for name in ("config.json","fold_scores.csv","repeat_scores.csv","summary.json"):
            if not (candidate/name).is_file():raise FileNotFoundError(candidate/name)
        dest.mkdir(parents=True,exist_ok=False)
        evidence=dest/"evidence";evidence.mkdir()
        for name in evidence_files:shutil.copy2(source/name,evidence/name)
        (evidence/"selected_candidate").mkdir()
        for name in ("config.json","fold_scores.csv","repeat_scores.csv","summary.json"):
            shutil.copy2(candidate/name,evidence/"selected_candidate"/name)
        self._write(dest/"base_config.json",context["base"].to_dict())
        frozen=dict(status="frozen_development_procedure",candidate_version=dest.name,
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),source_run=str(source),source_candidate=candidate_id,
            source_trial=source_config.source_trial_number,k=int(k),
            estimator_parameters=context["parameters"],feature_order=context["features"],
            training_sha256=self._sha(context["base"].train_path),
            ranking_method="Full-feature training-fitted booster total_gain; zero fill; ties by original column order.",
            selection_order="Preserve original column order after top-k selection.",
            refit_procedure="Fit ranking model on all labelled rows, select k, fit a fresh classifier on selected columns.",
            development_mean_auc=float(row["mean_auc"]),development_repeat_sd=float(row["std_auc"]),
            development_difference_vs_full_trial20=float(row["mean_difference_vs_trial20_control"]),
            score_is_independent_final_evaluation=False,versions=self._versions(),
            expected_test_rows=19750,id_column=context["base"].id_column,target_column=context["base"].target_column,
            default_test_path=str(self._path("data/raw/test.csv")),
            default_template_path=str(self._path("data/raw/sample_submission.csv")),
            policy="No retuning/reselection from global CV frequencies. Changed procedure requires a new candidate version.")
        self._write(dest/"frozen_candidate.json",frozen)
        checksums={str(p.relative_to(dest)):self._sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}
        self._write(dest/"snapshot_sha256.json",checksums)
        return dest

    def _verify_snapshot(self,frozen_dir):
        root=self._path(frozen_dir)
        expected=json.loads((root/"snapshot_sha256.json").read_text())
        if "frozen_candidate.json" not in expected or "base_config.json" not in expected:
            raise ValueError("Incomplete snapshot manifest.")
        for relative,digest in expected.items():
            p=(root/relative).resolve()
            if not p.is_relative_to(root):raise ValueError("Unsafe snapshot path.")
            if self._sha(p)!=digest:raise ValueError(f"Frozen snapshot changed: {relative}.")
        candidate=json.loads((root/"frozen_candidate.json").read_text())
        if candidate.get("status")!="frozen_development_procedure":raise ValueError("Invalid frozen status.")
        return root,candidate

    @staticmethod
    def _read_csv(path,id_column):
        with Path(path).open(encoding="utf-8-sig",newline="") as stream:
            header=next(csv.reader(stream),[])
        if not header or len(header)!=len(set(header)) or any(not s.strip() for s in header):
            raise ValueError("Empty/duplicate/blank CSV headers.")
        return pd.read_csv(path,dtype={id_column:"string"})

    def _test_inputs(self,test_path,template_path,features,id_column,target_column,expected_rows):
        test=self._read_csv(test_path,id_column);template=self._read_csv(template_path,id_column)
        if set(test.columns)!={id_column,*features} or len(test.columns)!=len(features)+1:
            raise ValueError("Test schema differs from training predictors plus ID.")
        if list(template.columns)!=[id_column,target_column]:raise ValueError("Template must have exactly id,target in that order.")
        if len(test)!=expected_rows or len(template)!=expected_rows:raise ValueError("Unexpected test/template row count.")
        for name,frame in (("test",test),("template",template)):
            ids=frame[id_column]
            if ids.isna().any() or not ids.is_unique or ids.str.strip().eq("").any():raise ValueError(f"Invalid {name} IDs.")
        if set(test[id_column])!=set(template[id_column]):raise ValueError("Test/template ID sets differ.")
        x=test[features].copy()
        if any(not pd.api.types.is_numeric_dtype(x[c]) for c in features) or not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("Test predictors must be finite numeric values; no fitting/imputation is allowed on test.")
        return x,test[id_column].copy(),template

    @staticmethod
    def _probabilities(model,x):
        if not np.array_equal(model.classes_,[0,1]):raise ValueError("Unexpected class orientation.")
        proba=np.asarray(model.predict_proba(x))
        if proba.shape!=(len(x),2) or not np.isfinite(proba).all() or not ((proba>=0)&(proba<=1)).all():
            raise ValueError("Invalid prediction probabilities.")
        if not np.allclose(proba.sum(axis=1),1.,atol=1e-6):raise ValueError("Probability row sums differ from one.")
        return proba[:,1].astype(float,copy=False)

    @staticmethod
    def _align(test_ids,scores,template,id_column,target_column):
        if len(test_ids)!=len(scores) or not test_ids.is_unique:raise ValueError("Prediction IDs/rows invalid.")
        by_id=pd.Series(scores,index=test_ids.to_numpy())
        result=template[[id_column]].copy()
        result[target_column]=result[id_column].map(by_id)
        values=result[target_column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():raise ValueError("Unmatched IDs or invalid aligned probabilities.")
        return result

    def refit_predict(self,frozen_dir,output_dir,test_path=None,template_path=None,dry_run=False):
        frozen,candidate=self._verify_snapshot(frozen_dir)
        output=self._path(output_dir)
        if output.exists():raise FileExistsError("Prediction output exists; choose a new directory.")
        versions=self._versions()
        if versions["xgboost"]!=candidate["versions"]["xgboost"]:
            raise ValueError("XGBoost version differs from the frozen environment; review before refitting.")
        from src.config.xgboost_experiment_config import XGBoostExperimentConfig
        from src.data.labelled_data_loader import LabelledDataLoader
        from src.features.gain_top_k_selector import GainTopKSelector
        base=XGBoostExperimentConfig.from_json(frozen/"base_config.json")
        if self._sha(base.train_path)!=candidate["training_sha256"]:raise ValueError("Training data changed after freeze.")
        x,y,train_ids,features=LabelledDataLoader().load(base)
        if features!=candidate["feature_order"]:raise ValueError("Training feature order differs from frozen record.")
        test_path=self._path(test_path or candidate["default_test_path"])
        template_path=self._path(template_path or candidate["default_template_path"])
        xtest,test_ids,template=self._test_inputs(test_path,template_path,features,candidate["id_column"],
                                                candidate["target_column"],candidate["expected_test_rows"])
        if dry_run:
            return dict(status="preflight_passed",train_rows=len(x),test_rows=len(xtest),k=candidate["k"],fits_performed=0)
        output.mkdir(parents=True,exist_ok=False)
        manifest=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),
            frozen_candidate=str(frozen),candidate_version=candidate["candidate_version"],versions=versions,
            train_rows=len(x),test_rows=len(xtest),k=candidate["k"],estimator_parameters=candidate["estimator_parameters"],
            input_sha256=dict(training=self._sha(base.train_path),test=self._sha(test_path),template=self._sha(template_path)),
            interpretation="Full-data refit and export, not an independent performance evaluation.",kaggle_upload_performed=False)
        path=output/"prediction_manifest.json";self._write(path,manifest)
        caught=[]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                params=candidate["estimator_parameters"]
                ranker=self.model_type(**params);ranker.fit(x,y)
                if ranker.get_booster().num_boosted_rounds()!=params["n_estimators"]:raise ValueError("Ranking round count differs.")
                selector=GainTopKSelector(k=candidate["k"]).fit_from_booster(ranker.get_booster(),features)
                selected=selector.get_feature_names_out().tolist()
                classifier=self.model_type(**params);classifier.fit(selector.transform(x),y)
                if classifier.get_booster().num_boosted_rounds()!=params["n_estimators"]:raise ValueError("Classifier round count differs.")
                original_scores=self._probabilities(classifier,selector.transform(xtest))
                ranker.save_model(output/"ranking_model.ubj")
                classifier.save_model(output/"classifier.ubj")
                loaded_ranker=self.model_type();loaded_ranker.load_model(output/"ranking_model.ubj")
                loaded_classifier=self.model_type();loaded_classifier.load_model(output/"classifier.ubj")
                loaded_selector=GainTopKSelector(k=candidate["k"]).fit_from_booster(loaded_ranker.get_booster(),features)
                if loaded_selector.get_feature_names_out().tolist()!=selected:raise ValueError("Reloaded ranking selects different columns.")
                scores=self._probabilities(loaded_classifier,loaded_selector.transform(xtest))
                if not np.allclose(scores,original_scores,rtol=0,atol=1e-12):raise ValueError("Reloaded model predictions differ.")
            warning_records=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
            self._write(output/"warnings.json",warning_records)
            if warning_records:raise RuntimeError("Captured warnings require review; submission CSV was not written.")
            self._write(output/"selected_features.json",dict(k=candidate["k"],feature_order=features,
                selected_features=selected,ranked_selected_features=selector.ranked_features(),
                source="New ranking fit using all 250 labelled rows; not global cross-fold selection frequencies."))
            frame=self._align(test_ids,scores,template,candidate["id_column"],candidate["target_column"])
            frame.to_csv(output/"submission.csv",index=False)
            exported=self._read_csv(output/"submission.csv",candidate["id_column"])
            if not exported[candidate["id_column"]].equals(template[candidate["id_column"]]):raise ValueError("Exported ID order mismatch.")
            if not np.allclose(exported[candidate["target_column"]],frame[candidate["target_column"]],rtol=0,atol=1e-12):
                raise ValueError("CSV round-trip probabilities differ.")
            for file,key in ((base.train_path,"training"),(test_path,"test"),(template_path,"template")):
                if self._sha(file)!=manifest["input_sha256"][key]:raise RuntimeError("An input file changed during refit/export.")
            manifest.update(status="ready_for_manual_submission",fits_performed=2,selected_features=selected,
                saved_and_reloaded_models_verified=True,submission_sha256=self._sha(output/"submission.csv"),
                submission_rows=len(frame),prediction_min=float(scores.min()),prediction_max=float(scores.max()),
                ranking_model_sha256=self._sha(output/"ranking_model.ubj"),classifier_sha256=self._sha(output/"classifier.ubj"))
            return dict(status=manifest["status"],submission=str(output/"submission.csv"),rows=len(frame),k=candidate["k"])
        except BaseException as exc:
            manifest.update(status="interrupted_or_failed",error=f"{type(exc).__name__}: {exc}")
            self._write(output/"prediction_error.json",dict(error=manifest["error"],traceback=traceback.format_exc()))
            self._write(output/"warnings.json",[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
            raise
        finally:
            manifest["finished_utc"]=datetime.now(timezone.utc).isoformat();self._write(path,manifest)

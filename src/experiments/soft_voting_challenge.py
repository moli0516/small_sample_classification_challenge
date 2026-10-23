"""Bounded L1 + frozen top-k XGBoost development study and prediction export."""
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class SoftVotingChallenge:
    def __init__(self):self.helper=None;self.model_type=None

    @staticmethod
    def blend(a,b):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if a.ndim!=1 or a.shape!=b.shape or len(a)==0:raise ValueError("Prediction shapes differ or are empty.")
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("Nonfinite predictions.")
        if ((a<0)|(a>1)|(b<0)|(b>1)).any():raise ValueError("Probabilities outside [0,1].")
        return .5*a+.5*b

    @staticmethod
    def _read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

    def _prepare(self,frozen_dir,l1_config,output_dir):
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        from src.config.experiment_config import ExperimentConfig
        from src.config.xgboost_experiment_config import XGBoostExperimentConfig
        from src.data.labelled_data_loader import LabelledDataLoader
        from src.features.gain_top_k_selector import GainTopKSelector
        from src.modeling.pipeline_factory import PipelineFactory
        from src.validation.saved_split_loader import SavedSplitLoader
        self.helper=FrozenXGBoostSubmission()
        h=self.helper
        output=h._path(output_dir)
        if output.exists():raise FileExistsError("Output exists; choose a new directory. Never overwrite historical results.")
        frozen,candidate=h._verify_snapshot(frozen_dir)
        versions=h._versions();self.model_type=h.model_type
        if versions["xgboost"]!=candidate["versions"]["xgboost"]:raise ValueError("XGBoost version differs from frozen candidate.")
        base=XGBoostExperimentConfig.from_json(frozen/"base_config.json")
        if h._sha(base.train_path)!=candidate["training_sha256"]:raise ValueError("Frozen training data changed.")
        x,y,ids,features=LabelledDataLoader().load(base)
        if features!=candidate["feature_order"]:raise ValueError("Feature order differs from frozen candidate.")
        path=h._path(l1_config);linear=ExperimentConfig.from_json(path)
        if linear.regularization_type!="l1" or linear.regularization_c!=1. or linear.feature_selection_k is not None:
            raise ValueError("This bounded study requires the existing all-feature L1 C=1 reference.")
        splits=SavedSplitLoader().load(frozen/"evidence/verified_split_indices.json",y,features)
        scores_path=frozen/"evidence/selected_candidate/fold_scores.csv"
        scores=pd.read_csv(scores_path)
        if len(scores)!=50 or scores.duplicated(["repeat","fold"]).any():raise ValueError("Invalid reference fold-score keys.")
        expected=scores.set_index(["repeat","fold"])["valid_auc"]
        if set(expected.index)!={(r["repeat"],r["fold"]) for r in splits} or not np.isfinite(expected).all():
            raise ValueError("Reference scores do not cover verified folds.")
        if not np.isclose(expected.groupby(level=0).mean().mean(),candidate["development_mean_auc"],rtol=0,atol=1e-12):
            raise ValueError("Reference scores disagree with frozen development mean.")
        sources=[base.train_path,path,frozen/"snapshot_sha256.json"]
        sources.extend(frozen/name for name in self._read(frozen/"snapshot_sha256.json"))
        for cls in (type(self),FrozenXGBoostSubmission,ExperimentConfig,XGBoostExperimentConfig,
                    LabelledDataLoader,GainTopKSelector,PipelineFactory,SavedSplitLoader):
            sources.append(Path(inspect.getfile(cls)).resolve())
        hashes={str(Path(p).resolve()):h._sha(p) for p in sources}
        return dict(output=output,frozen=frozen,candidate=candidate,versions=versions,base=base,
                    x=x,y=y,ids=ids,features=features,linear=linear,l1_path=path,
                    splits=splits,expected=expected,hashes=hashes)

    def _unchanged(self,c):
        for path,digest in c["hashes"].items():
            if self.helper._sha(path)!=digest:raise RuntimeError(f"Input/code changed: {path}")

    def _linear_model(self,config):
        from src.modeling.pipeline_factory import PipelineFactory
        return PipelineFactory().create(config)

    def _fit(self,c,x,y):
        from src.features.gain_top_k_selector import GainTopKSelector
        params=c["candidate"]["estimator_parameters"]
        linear=self._linear_model(c["linear"]);linear.fit(x,y)
        ranker=self.model_type(**params);ranker.fit(x,y)
        selector=GainTopKSelector(k=c["candidate"]["k"]).fit_from_booster(ranker.get_booster(),c["features"])
        tree=self.model_type(**params);tree.fit(selector.transform(x),y)
        for model in (ranker,tree):
            if model.get_booster().num_boosted_rounds()!=params["n_estimators"]:raise ValueError("Boosting-round count changed.")
        return linear,ranker,selector,tree

    def _predict(self,models,x):
        linear,ranker,selector,tree=models
        a=self.helper._probabilities(linear,x)
        b=self.helper._probabilities(tree,selector.transform(x))
        return dict(l1=a,xgboost=b,ensemble=self.blend(a,b))

    @staticmethod
    def _summary(folds):
        repeats=folds.groupby(["repeat","split_seed"],as_index=False)[["l1","xgboost","ensemble"]].mean()
        repeats["ensemble_minus_xgboost"]=repeats.ensemble-repeats.xgboost
        repeats["ensemble_minus_l1"]=repeats.ensemble-repeats.l1
        d=repeats.ensemble_minus_xgboost
        summary=dict(mean_auc={key:float(repeats[key].mean()) for key in ("l1","xgboost","ensemble")},
            repeat_sd={key:float(repeats[key].std(ddof=1)) for key in ("l1","xgboost","ensemble")},
            mean_difference_vs_xgboost=float(d.mean()),wins_vs_xgboost=int((d>1e-12).sum()),
            losses_vs_xgboost=int((d< -1e-12).sum()),ties_vs_xgboost=int((abs(d)<=1e-12).sum()))
        summary["promote_ensemble"]=bool(d.mean()>1e-12 and (d>1e-12).sum()>=6)
        summary["promotion_rule"]="Positive mean paired difference vs XGBoost and wins in at least 6/10 repeats; pragmatic rule, not significance test."
        summary["interpretation"]="Adaptive development comparison on previously used splits, not independent final evaluation."
        return repeats,summary

    def cv(self,frozen_dir,l1_config,output_dir,dry_run=False):
        c=self._prepare(frozen_dir,l1_config,output_dir);h=self.helper
        if dry_run:return dict(status="preflight_passed",folds=len(c["splits"]),planned_fits=3*len(c["splits"]),test_opened=False)
        out=c["output"];out.mkdir(parents=True,exist_ok=False)
        h._write(out/"l1_config.json",c["linear"].to_dict())
        settings=dict(frozen_dir=str(c["frozen"]),l1_config_path=str(c["l1_path"]),weights=[.5,.5],
            k=c["candidate"]["k"],xgboost_parameters=c["candidate"]["estimator_parameters"],versions=c["versions"],
            input_and_code_sha256=c["hashes"],test_opened=False,planned_fits=3*len(c["splits"]))
        h._write(out/"config.json",settings)
        manifest=dict(status="running",completed_folds=0,started_utc=datetime.now(timezone.utc).isoformat(),
            fits_per_completed_fold=3,test_opened=False,kaggle_upload_performed=False)
        rows,predictions,all_warnings=[],[],[]
        try:
            for split in c["splits"]:
                self._unchanged(c)
                tr,va=split["train_indices"],split["valid_indices"]
                caught=[]
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        models=self._fit(c,c["x"].iloc[tr],c["y"].iloc[tr])
                        probs=self._predict(models,c["x"].iloc[va])
                        auc={name:float(roc_auc_score(c["y"].iloc[va],p)) for name,p in probs.items()}
                finally:
                    all_warnings.extend(dict(repeat=split["repeat"],fold=split["fold"],category=w.category.__name__,message=str(w.message)) for w in caught)
                    h._write(out/"warnings.json",all_warnings)
                if caught:raise RuntimeError("Python warnings require review; CV stopped.")
                expected=float(c["expected"].loc[(split["repeat"],split["fold"])])
                if not np.isclose(auc["xgboost"],expected,rtol=0,atol=1e-12):
                    raise RuntimeError("Top-k XGBoost control does not reproduce; do not relax tolerance.")
                key={name:split[name] for name in ("repeat","split_seed","fold")}
                rows.append(dict(key,**auc,reference_xgboost=expected,control_matches=True))
                for pos,index in enumerate(va):
                    predictions.append(dict(key,row_index=int(index),id=str(c["ids"].iloc[index]),
                        target=int(c["y"].iloc[index]),**{name:float(p[pos]) for name,p in probs.items()}))
                pd.DataFrame(rows).to_csv(out/"fold_scores.csv",index=False)
                pd.DataFrame(predictions).to_csv(out/"validation_probabilities.csv",index=False)
                manifest["completed_folds"]=len(rows);h._write(out/"run_manifest.json",manifest)
                print(f"[{len(rows)}/{len(c['splits'])}] repeat={key['repeat']} fold={key['fold']} ensemble={auc['ensemble']:.6f}",flush=True)
            self._unchanged(c)
            repeated,summary=self._summary(pd.DataFrame(rows))
            repeated.to_csv(out/"repeat_scores.csv",index=False);h._write(out/"summary.json",summary)
            manifest.update(status="completed",fits_performed=3*len(rows),control_auc_reproduced=True,
                            promote_ensemble=summary["promote_ensemble"])
            artifacts={p.name:h._sha(p) for p in out.iterdir() if p.is_file() and p.name!="run_manifest.json"}
            manifest["artifact_sha256"]=artifacts
            return summary
        except BaseException as exc:
            manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
            h._write(out/"error.json",dict(error=manifest["error"],traceback=traceback.format_exc()));raise
        finally:
            manifest["finished_utc"]=datetime.now(timezone.utc).isoformat();h._write(out/"run_manifest.json",manifest)

    def predict(self,cv_run,output_dir,test_path=None,template_path=None,dry_run=False):
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        helper=FrozenXGBoostSubmission();run=helper._path(cv_run)
        record=self._read(run/"run_manifest.json")
        if record.get("status")!="completed" or record.get("completed_folds")!=50 or not record.get("control_auc_reproduced"):
            raise ValueError("Need a completed 50-fold ensemble study with reproduced XGBoost control.")
        for name,digest in record["artifact_sha256"].items():
            path=(run/name).resolve()
            if not path.is_relative_to(run) or helper._sha(path)!=digest:raise ValueError("CV artifacts changed.")
        settings=self._read(run/"config.json");summary=self._read(run/"summary.json")
        if not summary.get("promote_ensemble"):raise ValueError("Ensemble did not meet the predeclared promotion rule; retain single-model candidate.")
        c=self._prepare(settings["frozen_dir"],settings["l1_config_path"],output_dir);h=self.helper
        if c["hashes"]!=settings["input_and_code_sha256"]:raise ValueError("Inputs/code differ from evaluated ensemble.")
        if c["versions"]!=settings["versions"]:raise ValueError("Environment differs from evaluated ensemble.")
        test=h._path(test_path or c["candidate"]["default_test_path"])
        template=h._path(template_path or c["candidate"]["default_template_path"])
        xtest,ids,template_frame=h._test_inputs(test,template,c["features"],c["base"].id_column,
                                                c["base"].target_column,c["candidate"]["expected_test_rows"])
        if dry_run:return dict(status="prediction_preflight_passed",test_rows=len(xtest),fits_performed=0)
        out=c["output"];out.mkdir(parents=True,exist_ok=False)
        manifest=dict(status="running",cv_run=str(run),weights=[.5,.5],versions=c["versions"],
            started_utc=datetime.now(timezone.utc).isoformat(),kaggle_upload_performed=False,
            input_sha256={str(p):h._sha(p) for p in (test,template)},interpretation="Full-data refit/export; not new validation.")
        caught=[]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                models=self._fit(c,c["x"],c["y"]);before=self._predict(models,xtest)
                linear,ranker,selector,tree=models
                joblib.dump(linear,out/"l1_pipeline.joblib")
                ranker.save_model(out/"ranking_model.ubj");tree.save_model(out/"xgb_classifier.ubj")
                loaded_linear=joblib.load(out/"l1_pipeline.joblib")
                loaded_ranker=self.model_type();loaded_ranker.load_model(out/"ranking_model.ubj")
                loaded_tree=self.model_type();loaded_tree.load_model(out/"xgb_classifier.ubj")
                from src.features.gain_top_k_selector import GainTopKSelector
                loaded_selector=GainTopKSelector(k=c["candidate"]["k"]).fit_from_booster(loaded_ranker.get_booster(),c["features"])
                if loaded_selector.get_feature_names_out().tolist()!=selector.get_feature_names_out().tolist():raise ValueError("Reloaded selection differs.")
                after=self._predict((loaded_linear,loaded_ranker,loaded_selector,loaded_tree),xtest)
                for name in before:
                    if not np.allclose(before[name],after[name],rtol=0,atol=1e-12):raise ValueError("Reloaded probabilities differ.")
            if caught:raise RuntimeError("Python warnings require review; no submission written.")
            self._unchanged(c)
            for path,digest in manifest["input_sha256"].items():
                if h._sha(path)!=digest:raise ValueError("Prediction input changed.")
            frame=h._align(ids,after["ensemble"],template_frame,c["base"].id_column,c["base"].target_column)
            frame.to_csv(out/"submission.csv",index=False)
            saved=h._read_csv(out/"submission.csv",c["base"].id_column)
            if not saved[c["base"].id_column].equals(template_frame[c["base"].id_column]):raise ValueError("CSV ID order mismatch.")
            if not np.allclose(saved[c["base"].target_column],frame[c["base"].target_column],rtol=0,atol=1e-12):raise ValueError("CSV probability round-trip mismatch.")
            h._write(out/"ensemble_metadata.json",dict(weights=[.5,.5],l1_config=c["linear"].to_dict(),
                original_feature_order=c["features"],xgboost_selected_features=selector.get_feature_names_out().tolist(),
                xgboost_parameters=c["candidate"]["estimator_parameters"],source_frozen_candidate=str(c["frozen"])))
            manifest.update(status="ready_for_manual_submission",rows=len(frame),fits_performed=3,
                saved_and_reloaded_models_verified=True,submission_sha256=h._sha(out/"submission.csv"))
            manifest["artifact_sha256"]={p.name:h._sha(p) for p in out.iterdir() if p.is_file()}
            return dict(status=manifest["status"],rows=len(frame),submission=str(out/"submission.csv"))
        except BaseException as exc:
            manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
            h._write(out/"error.json",dict(error=manifest["error"],traceback=traceback.format_exc()));raise
        finally:
            h._write(out/"warnings.json",[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
            manifest["finished_utc"]=datetime.now(timezone.utc).isoformat();h._write(out/"prediction_manifest.json",manifest)

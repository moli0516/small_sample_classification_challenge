"""Evaluate fold-local top-k selection against the reproduced Trial 20 control."""
from datetime import datetime,timezone
import hashlib
import inspect
import json
import logging
import math
from pathlib import Path
import platform
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from src.features.gain_top_k_selector import GainTopKSelector


class XGBoostFeatureSelectionRunner:
    AUC_TOLERANCE=1e-12

    def __init__(self,factory_type=None,result_type=None):
        self.factory_type=factory_type
        self.result_type=result_type
        self.logger=logging.getLogger(__name__)

    @staticmethod
    def _sha(path):
        h=hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda:stream.read(1048576),b""):
                h.update(block)
        return h.hexdigest()

    @classmethod
    def _clean(cls,obj):
        if isinstance(obj,dict):return {str(k):cls._clean(v) for k,v in obj.items()}
        if isinstance(obj,(list,tuple,np.ndarray)):return [cls._clean(v) for v in obj]
        if isinstance(obj,np.generic):return cls._clean(obj.item())
        if isinstance(obj,Path):return str(obj)
        if isinstance(obj,float) and not math.isfinite(obj):return None
        return obj

    @classmethod
    def _save_json(cls,path,obj):
        Path(path).write_text(json.dumps(cls._clean(obj),indent=2,ensure_ascii=False,
                                         allow_nan=False)+"\n",encoding="utf-8")

    def _prepare(self,config,confirmed):
        if not confirmed:
            raise ValueError("Confirm unchanged data/order/metric using --confirm-reference-alignment.")
        if config.output_dir.exists():
            raise FileExistsError("Output already exists; choose a new directory.")
        from src.config.xgboost_experiment_config import XGBoostExperimentConfig
        from src.data.labelled_data_loader import LabelledDataLoader
        from src.modeling.xgboost_pipeline_factory import XGBoostPipelineFactory
        from src.validation.saved_split_loader import SavedSplitLoader
        from src.validation.paired_comparison import PairedComparison
        from src.validation.validation_result import ValidationResult
        if self.factory_type is None:self.factory_type=XGBoostPipelineFactory
        if self.result_type is None:self.result_type=ValidationResult
        version=self.factory_type.check_environment()
        study=config.source_study_dir
        source_manifest=json.loads((study/"run_manifest.json").read_text())
        if source_manifest.get("status")!="completed":
            raise ValueError("Source study is not complete.")
        best=json.loads((study/"best_parameters.json").read_text())
        if best["trial_number"]!=config.source_trial_number:
            raise ValueError("Source selected trial differs from requested trial.")
        trial=study/"trials"/f"trial_{config.source_trial_number:04d}"
        source_comparison=pd.read_csv(trial/"e5b_comparison.csv")
        if len(source_comparison)!=1 or not bool(source_comparison.iloc[0]["eligible"]):
            raise ValueError("Expected one eligible source candidate.")
        candidate_id=str(source_comparison.iloc[0]["experiment_id"])
        if Path(candidate_id).name!=candidate_id:
            raise ValueError("Unsafe source candidate identifier.")
        candidate=trial/candidate_id
        parameter_path=candidate/"config.json"
        parameters=json.loads(parameter_path.read_text())["estimator_parameters"]
        for name,value in best["parameters"].items():
            if parameters.get(name)!=value:raise ValueError(f"Source parameter mismatch: {name}.")
        base=XGBoostExperimentConfig.from_json(study/"base_config.json")
        overrides={name:parameters[name] for name in
                   ("learning_rate","min_child_weight","reg_lambda","subsample","colsample_bytree")}
        factory=self.factory_type(overrides=overrides)
        actual=factory.parameters(base,parameters["max_depth"],parameters["n_estimators"])
        if actual!=parameters:
            raise ValueError("Current factory cannot exactly reproduce source constructor parameters.")
        trial_manifest_path=trial/"run_manifest.json"
        trial_manifest=json.loads(trial_manifest_path.read_text())
        if trial_manifest.get("status")!="completed" or trial_manifest.get("completed_fits")!=50:
            raise ValueError("Source trial is incomplete.")
        for name in ("train_path","split_indices_path","sparse_split_indices_path","reference_scores_path"):
            if self._sha(getattr(base,name))!=trial_manifest["input_sha256"][name]:
                raise ValueError(f"Historical input hash mismatch: {name}.")
        x,y,ids,features=LabelledDataLoader().load(base)
        if any(k>=len(features) for k in config.k_values):
            raise ValueError("Selected k must be smaller than the full feature count; control is added automatically.")
        loader=SavedSplitLoader()
        splits=loader.load(study/"verified_split_indices.json",y,features)
        original=loader.load(base.split_indices_path,y,features)
        sparse=loader.load(base.sparse_split_indices_path,y,features)
        if loader.signature(splits)!=loader.signature(original) or loader.signature(splits)!=loader.signature(sparse):
            raise ValueError("Saved split assignments disagree.")
        original_scores=self.result_type(pd.read_csv(candidate/"fold_scores.csv"))
        expected=original_scores.fold_scores.set_index(["repeat","fold"])["valid_auc"]
        if len(expected)!=50:raise ValueError("Expected 50 original fold scores.")
        if not np.isclose(original_scores.summary()["mean_auc"],best["mean_auc"],rtol=0,atol=self.AUC_TOLERANCE):
            raise ValueError("Source fold-score aggregate differs from best_parameters mean AUC.")
        refs=PairedComparison().load_references(base.reference_scores_path)
        sources=[study/"run_manifest.json",study/"best_parameters.json",study/"base_config.json",
                 study/"verified_split_indices.json",parameter_path,candidate/"fold_scores.csv",trial_manifest_path]
        sources += [getattr(base,n) for n in ("train_path","split_indices_path","sparse_split_indices_path","reference_scores_path")]
        for cls in (type(config),GainTopKSelector,XGBoostFeatureSelectionRunner,self.factory_type,
                    self.result_type,SavedSplitLoader,PairedComparison,LabelledDataLoader):
            sources.append(Path(inspect.getfile(cls)).resolve())
        hashes={str(Path(p).resolve()):self._sha(p) for p in sources}
        return dict(x=x,y=y,ids=ids,features=features,splits=splits,expected=expected,references=refs,
                    base=base,factory=factory,parameters=parameters,hashes=hashes,
                    source_mean=best["mean_auc"],source_candidate=candidate_id,
                    split_signature=loader.signature(splits),xgboost_version=version,
                    source_versions=trial_manifest.get("versions",{}))

    def _check_unchanged(self,context):
        for path,value in context["hashes"].items():
            if self._sha(path)!=value:
                raise RuntimeError(f"Input/source changed during experiment: {path}.")

    def _fit(self,context,split,k,name,xtrain,xvalid,selector_review=False):
        key={name:split[name] for name in ("repeat","split_seed","fold")}
        key.update(experiment_id=name,k=k)
        tr,va=split["train_indices"],split["valid_indices"]
        record=dict(key,n_train=len(tr),n_validation=len(va),valid=False,auc=None,error="",
                    selector_review_required=bool(selector_review),fit_seconds=None)
        caught,predictions,model,booster_config=[],[],None,None
        started=time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                params=context["parameters"]
                model=context["factory"].create(context["base"],params["max_depth"],params["n_estimators"])
                model.fit(xtrain,context["y"].iloc[tr])
                record["fit_seconds"]=time.perf_counter()-started
                if not np.array_equal(model.classes_,[0,1]):raise ValueError("Unexpected class order.")
                probabilities=np.asarray(model.predict_proba(xvalid))
                if probabilities.shape!=(len(va),2) or not np.isfinite(probabilities).all():
                    raise ValueError("Invalid probability shape/values.")
                if not ((probabilities>=0)&(probabilities<=1)).all() or not np.allclose(probabilities.sum(axis=1),1.,atol=1e-6):
                    raise ValueError("Invalid probability range/sums.")
                booster=model.named_steps["classifier"].get_booster()
                rounds=int(booster.num_boosted_rounds())
                if rounds!=params["n_estimators"]:raise ValueError("Unexpected boosting-round count.")
                scores=probabilities[:,1]
                record.update(valid=True,auc=float(roc_auc_score(context["y"].iloc[va],scores)),rounds_fitted=rounds,
                              features_used=len(booster.get_score(importance_type="weight")),
                              unique_scores=int(np.unique(scores).size))
                booster_config=json.loads(booster.save_config())
                predictions=[dict(key,row_index=int(i),id=str(context["ids"].iloc[i]),
                    target=int(context["y"].iloc[i]),probability_target_1=float(score)) for i,score in zip(va,scores)]
        except Exception as exc:
            record.update(valid=False,error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        if record["fit_seconds"] is None:record["fit_seconds"]=time.perf_counter()-started
        record["total_seconds"]=time.perf_counter()-started
        record["warning_count"]=len(caught)
        record["warnings_json"]=json.dumps([dict(category=w.category.__name__,message=str(w.message)) for w in caught])
        record["review_required"]=bool(caught or selector_review or not record["valid"])
        for pred in predictions:pred["fold_valid"]=record["valid"]
        return record,predictions,model,booster_config

    @staticmethod
    def _candidates(config,n_features):
        return [(f"{config.experiment_id}_all_features",n_features)]+[(f"{config.experiment_id}_k_{k}",k) for k in config.k_values]

    def _summarize(self,config,context,records):
        folds=pd.DataFrame(records)
        rows,all_repeats=[],[]
        control_name=self._candidates(config,len(context["features"]))[0][0]
        control=folds.loc[folds["experiment_id"]==control_name]
        control_repeats=control.groupby(["repeat","split_seed"],as_index=False).agg(control_mean_auc=("auc","mean"))
        for name,k in self._candidates(config,len(context["features"])):
            f=folds.loc[folds["experiment_id"]==name]
            complete=len(f)==50 and bool(f["valid"].all())
            review=bool(f["review_required"].any())
            row=dict(experiment_id=name,k=k,completed_folds=len(f),valid_folds=int(f["valid"].sum()),
                failed_folds=int((~f["valid"]).sum()),warning_folds=int((f["warning_count"]>0).sum()),
                selector_review_folds=int(f["selector_review_required"].sum()),complete_valid=complete,
                review_required=review,eligible=bool(complete and not review),total_fit_seconds=float(f["fit_seconds"].sum()))
            if complete:
                scores=f.rename(columns={"n_train":"train_rows","n_validation":"valid_rows","auc":"valid_auc"})
                result=self.result_type(scores)
                result.save(config.output_dir/name)
                row.update(result.summary())
                repeated=result.repeat_scores().merge(control_repeats,on=["repeat","split_seed"],validate="one_to_one")
                if len(repeated)!=10 or repeated["control_mean_auc"].isna().any():raise RuntimeError("Incomplete paired control.")
                difference=repeated["mean_auc"]-repeated["control_mean_auc"]
                repeated["difference_vs_trial20_control"]=difference
                row.update(mean_difference_vs_trial20_control=float(difference.mean()),
                    sd_difference_vs_trial20_control=float(difference.std(ddof=1)),
                    min_difference_vs_trial20_control=float(difference.min()),
                    max_difference_vs_trial20_control=float(difference.max()),
                    wins_vs_trial20_control=int((difference>1e-12).sum()),losses_vs_trial20_control=int((difference< -1e-12).sum()),
                    ties_vs_trial20_control=int((abs(difference)<=1e-12).sum()))
                # L1 references are joined explicitly without refitting either L1 model.
                repeated=repeated.merge(context["references"],on=["repeat","split_seed"],validate="one_to_one")
                for label in ("e4_leader","e4_sparse"):
                    d=repeated["mean_auc"]-repeated[f"{label}_auc"]
                    repeated[f"difference_vs_{label}"]=d
                    row.update({f"mean_difference_vs_{label}":float(d.mean()),
                        f"sd_difference_vs_{label}":float(d.std(ddof=1)),
                        f"wins_vs_{label}":int((d>1e-12).sum()),f"losses_vs_{label}":int((d< -1e-12).sum()),
                        f"ties_vs_{label}":int((abs(d)<=1e-12).sum())})
                repeated["experiment_id"],repeated["k"]=name,k
                repeated["review_required"]=review
                all_repeats.extend(repeated.to_dict("records"))
            rows.append(row)
        table=pd.DataFrame(rows)
        table.to_csv(config.output_dir/"feature_selection_comparison.csv",index=False)
        pd.DataFrame(all_repeats).to_csv(config.output_dir/"feature_selection_paired_repeat_scores.csv",index=False)
        eligible=table.loc[table["eligible"]]
        if eligible.empty:
            decision=dict(status="no_eligible_candidate",selected_experiment_id=None,promote_over_trial20=False)
        else:
            best=eligible["mean_auc"].max()
            winner=eligible.loc[abs(eligible["mean_auc"]-best)<=1e-12].sort_values("k").iloc[0]
            decision=dict(status="development_selection_only",selected_experiment_id=winner["experiment_id"],k=int(winner["k"]),
                mean_auc=float(winner["mean_auc"]),mean_difference_vs_trial20_control=float(winner["mean_difference_vs_trial20_control"]),
                promote_over_trial20=bool(float(winner["mean_difference_vs_trial20_control"])>1e-12),
                note="Fold-local selection, but parameters/k decisions remain adaptive development choices. Not independent evaluation.")
        self._save_json(config.output_dir/"feature_selection_decision.json",decision)
        return table

    def run(self,config,dry_run=False,confirm_reference_alignment=False):
        context=self._prepare(config,confirm_reference_alignment)
        output=config.output_dir
        output.mkdir(parents=True,exist_ok=False)
        self._save_json(output/"config.json",config.to_dict())
        self._save_json(output/"source_trial_parameters.json",context["parameters"])
        self._save_json(output/"verified_split_indices.json",dict(index_type="zero_based_row_position",
            n_rows=len(context["x"]),n_splits=5,split_seeds=list(range(10)),feature_names=context["features"],splits=context["splits"]))
        context["references"].to_csv(output/"reference_repeat_scores.csv",index=False)
        manifest=dict(status="validated",dry_run=bool(dry_run),started_utc=datetime.now(timezone.utc).isoformat(),
            source_trial=config.source_trial_number,source_candidate=context["source_candidate"],source_mean_auc=context["source_mean"],
            source_versions=context.get("source_versions",{}),
            split_signature=context["split_signature"],input_and_source_sha256=context["hashes"],
            versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,xgboost=context["xgboost_version"]),
            planned_fits=50*(1+len(config.k_values)),candidate_k=[len(context["features"]),*config.k_values],
            ranking="Training-fold control booster total_gain only; zero-fill; ties by original column order.",
            transform_order="Original predictor order after selecting top-k.",
            control_reuse="Control booster also supplies ranking; no extra ranking fit.",
            control_auc_tolerance=self.AUC_TOLERANCE,early_stopping=False,pruning=False,
            warning_policy="Any Python warning or warning-bearing ranking model requires review; no automatic eligibility.",
            metric="mean_fold_auc_per_repeat",interpretation="Adaptive development study, not independent final evaluation.")
        path=output/"run_manifest.json"
        self._save_json(path,manifest)
        if dry_run:
            self.logger.info("Feature-selection preflight passed. No model fits. Use a new full-run output directory.")
            return 0
        all_records,checks,selection_records,rank_records=[],[],[],[]
        candidates=self._candidates(config,len(context["features"]))
        local={name:[] for name,k in candidates}
        predictions={name:[] for name,k in candidates}
        for name,k in candidates:
            dest=output/name;dest.mkdir()
            self._save_json(dest/"config.json",dict(experiment_id=name,k=k,estimator_parameters=context["parameters"],
                selector="none" if k==len(context["features"]) else "training_fold_control_total_gain_top_k"))
        manifest["status"]="running";self._save_json(path,manifest)
        def checkpoint(rec,pred,booster):
            name=rec["experiment_id"]
            all_records.append(rec);local[name].append(rec);predictions[name].extend(pred)
            pd.DataFrame(all_records).to_csv(output/"fold_diagnostics.csv",index=False)
            pd.DataFrame(local[name]).to_csv(output/name/"fold_diagnostics.csv",index=False)
            columns=["repeat","split_seed","fold","experiment_id","k","row_index","id","target","probability_target_1","fold_valid"]
            pd.DataFrame(predictions[name],columns=columns).to_csv(output/name/"validation_probabilities.csv",index=False)
            if booster is not None:
                with (output/name/"booster_configs.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(self._clean(dict(repeat=rec["repeat"],fold=rec["fold"],config=booster)),allow_nan=False)+"\n")
            self.logger.info("[%d/%d] k=%d repeat=%d fold=%d AUC=%s valid=%s",len(all_records),manifest["planned_fits"],
                rec["k"],rec["repeat"],rec["fold"],rec["auc"],rec["valid"])
        try:
            for split in context["splits"]:
                self._check_unchanged(context)
                tr,va=split["train_indices"],split["valid_indices"]
                train,valid=context["x"].iloc[tr],context["x"].iloc[va]
                control_name,full_k=candidates[0]
                rec,pred,control,booster=self._fit(context,split,full_k,control_name,train,valid)
                checkpoint(rec,pred,booster)
                if not rec["valid"]:
                    raise RuntimeError("Control fit invalid; cannot obtain a valid training-only ranking. Inspect diagnostics.")
                original=float(context["expected"].loc[(split["repeat"],split["fold"])])
                matches=bool(np.isclose(rec["auc"],original,rtol=0,atol=self.AUC_TOLERANCE))
                checks.append(dict(repeat=split["repeat"],fold=split["fold"],original_auc=original,reproduced_auc=rec["auc"],matches=matches))
                pd.DataFrame(checks).to_csv(output/"control_auc_reproduction.csv",index=False)
                if not matches:raise RuntimeError("Control AUC does not reproduce. Experiment stopped; do not relax tolerance to bypass it.")
                selector_booster=control.named_steps["classifier"].get_booster()
                for name,k in candidates[1:]:
                    selector=GainTopKSelector(k=k).fit_from_booster(selector_booster,context["features"])
                    selection_records.append(dict(repeat=split["repeat"],split_seed=split["split_seed"],fold=split["fold"],k=k,
                        selected_features=selector.get_feature_names_out().tolist(),ranked_selected_features=selector.ranked_features(),
                        selector_review_required=rec["review_required"]))
                    if k==config.k_values[0]:
                        for rank,index in enumerate(selector.ranking_indices_,1):
                            rank_records.append(dict(repeat=split["repeat"],fold=split["fold"],rank=rank,
                                feature=context["features"][index],training_total_gain=float(selector.gains_[index])))
                    selected_train,selected_valid=selector.transform(train),selector.transform(valid)
                    sub_record,sub_pred,_,sub_booster=self._fit(context,split,k,name,selected_train,selected_valid,
                                                               selector_review=rec["review_required"])
                    checkpoint(sub_record,sub_pred,sub_booster)
                self._save_json(output/"selected_features.json",selection_records)
                pd.DataFrame(rank_records).to_csv(output/"training_feature_rankings.csv",index=False)
                self._check_unchanged(context)
            table=self._summarize(config,context,all_records)
            frequencies=[]
            for k in config.k_values:
                selected=[r for r in selection_records if r["k"]==k]
                for feature in context["features"]:
                    count=sum(feature in r["selected_features"] for r in selected)
                    frequencies.append(dict(k=k,feature=feature,selected_folds=count,total_folds=len(selected),selection_frequency=count/len(selected)))
            pd.DataFrame(frequencies).to_csv(output/"selection_frequencies.csv",index=False)
            manifest.update(status="completed",completed_fits=len(all_records),control_auc_reproduced=all(c["matches"] for c in checks),
                            eligible_candidates=int(table["eligible"].sum()))
            return 0 if bool(table["eligible"].all()) else 2
        except BaseException as exc:
            manifest.update(status="interrupted_or_failed",completed_fits=len(all_records),error=f"{type(exc).__name__}: {exc}")
            self._save_json(output/"experiment_error.json",dict(error=manifest["error"],traceback=traceback.format_exc()))
            raise
        finally:
            manifest["finished_utc"]=datetime.now(timezone.utc).isoformat()
            self._save_json(path,manifest)

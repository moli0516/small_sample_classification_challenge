"""Bounded Optuna search reusing the integrated E5b runner and saved splits."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd
import sklearn
from src.config.xgboost_optuna_config import XGBoostOptunaConfig
from src.modeling.xgboost_pipeline_factory import XGBoostPipelineFactory
from src.experiments.optuna_trial_rejected import OptunaTrialRejected


class XGBoostOptunaRunner:
    """No pruning, early stopping, resume, parallel optimization or public-score tuning."""

    def __init__(self, runner_type=None, factory_type=None, optuna_module=None):
        self.runner_type=runner_type
        self.factory_type=factory_type or XGBoostPipelineFactory
        self.optuna_module=optuna_module
        self.logger=logging.getLogger(__name__)

    @staticmethod
    def _sha256(path):
        h=hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda:stream.read(1048576),b""):
                h.update(block)
        return h.hexdigest()

    @classmethod
    def _jsonable(cls,obj):
        if isinstance(obj,dict):
            return {str(k):cls._jsonable(v) for k,v in obj.items()}
        if isinstance(obj,(list,tuple,np.ndarray)):
            return [cls._jsonable(v) for v in obj]
        if isinstance(obj,np.generic):
            return cls._jsonable(obj.item())
        if isinstance(obj,Path):
            return str(obj)
        if isinstance(obj,float) and not math.isfinite(obj):
            return None
        return obj

    @classmethod
    def _save_json(cls,path,obj):
        Path(path).write_text(json.dumps(cls._jsonable(obj),indent=2,ensure_ascii=False,
                                        allow_nan=False)+"\n",encoding="utf-8")

    def _dependencies(self):
        if self.optuna_module is None:
            import optuna
            self.optuna_module=optuna
        if self.runner_type is None:
            from src.experiments.xgboost_experiment_runner import XGBoostExperimentRunner
            self.runner_type=XGBoostExperimentRunner
        return self.optuna_module

    def _prepare(self,config,confirmed):
        if not confirmed:
            raise ValueError("Confirm unchanged historical data, row order and metric using --confirm-reference-alignment.")
        if config.output_dir.exists():
            raise FileExistsError(f"Output already exists: {config.output_dir}. No overwrite or automatic resume.")
        optuna=self._dependencies()
        xgboost_version=self.factory_type.check_environment()
        from src.config.xgboost_experiment_config import XGBoostExperimentConfig
        from src.data.labelled_data_loader import LabelledDataLoader
        from src.validation.saved_split_loader import SavedSplitLoader
        from src.validation.paired_comparison import PairedComparison
        base=XGBoostExperimentConfig.from_json(config.base_config_path)
        x,y,ids,features=LabelledDataLoader().load(base)
        loader=SavedSplitLoader()
        splits=loader.load(base.split_indices_path,y,features)
        sparse=loader.load(base.sparse_split_indices_path,y,features)
        if loader.signature(splits)!=loader.signature(sparse):
            raise ValueError("The two L1 references have different saved split assignments.")
        references=PairedComparison().load_references(base.reference_scores_path)
        sources=[config.base_config_path,base.train_path,base.split_indices_path,
                 base.sparse_split_indices_path,base.reference_scores_path]
        classes=(type(config),type(base),self.factory_type,self.runner_type,
                 SavedSplitLoader,PairedComparison,LabelledDataLoader,XGBoostOptunaRunner)
        for cls in classes:
            path=Path(inspect.getfile(cls)).resolve()
            if path.is_file():
                sources.append(path)
        cli=config.project_root()/"scripts/run_e5b_optuna.py"
        if cli.is_file():
            sources.append(cli)
        hashes={str(Path(p).resolve()):self._sha256(p) for p in sources}
        return dict(base=base,splits=splits,references=references,features=features,hashes=hashes,
                    split_signature=loader.signature(splits),
                    xgboost_version=xgboost_version,optuna_version=getattr(optuna,"__version__","unknown"))

    def _check_unchanged(self,context):
        for path,expected in context["hashes"].items():
            if self._sha256(path)!=expected:
                raise RuntimeError(f"Input/code changed during Optuna study: {path}. Study stopped.")

    @staticmethod
    def _suggest(trial,space):
        result={}
        for name,spec in space.items():
            if spec["type"]=="int":
                result[name]=trial.suggest_int(name,spec["low"],spec["high"])
            elif spec["type"]=="categorical":
                result[name]=trial.suggest_categorical(name,spec["choices"])
            else:
                result[name]=trial.suggest_float(name,spec["low"],spec["high"],log=spec["log"])
        return result

    def _objective(self,trial,config,context):
        self._check_unchanged(context)
        parameters=self._suggest(trial,config.search_space)
        output=config.output_dir/"trials"/f"trial_{trial.number:04d}"
        trial_config=replace(context["base"],
            experiment_id=f"{config.experiment_id}_trial_{trial.number:04d}",
            output_dir=output,max_depth_values=(int(parameters["max_depth"]),),
            n_estimators_values=(int(parameters["n_estimators"]),))
        overrides={k:v for k,v in parameters.items() if k not in ("max_depth","n_estimators")}
        trial.set_user_attr("artifact_directory",str(output))
        trial.set_user_attr("planned_folds",50)
        trial.set_user_attr("requested_parameters",self._jsonable(parameters))
        code=self.runner_type(factory=self.factory_type(overrides=overrides)).run(
            trial_config,confirm_reference_alignment=True)
        self._check_unchanged(context)
        manifest=json.loads((output/"run_manifest.json").read_text(encoding="utf-8"))
        table=pd.read_csv(output/"e5b_comparison.csv")
        if len(table)!=1:
            raise RuntimeError("Trial runner must produce exactly one candidate row.")
        row=table.iloc[0]
        for field in ("completed_folds","valid_folds","warning_folds","failed_folds"):
            trial.set_user_attr(field,int(row[field]))
        if manifest.get("status")!="completed":
            raise RuntimeError("Trial runner did not finish normally; study stopped.")
        valid=(manifest.get("completed_fits")==50 and int(row["completed_folds"])==50
               and int(row["valid_folds"])==50 and int(row["failed_folds"])==0
               and int(row["warning_folds"])==0 and bool(row["eligible"]) and code==0)
        if not valid:
            reason="Trial rejected: not all 50 folds eligible, or captured warnings require review."
            trial.set_user_attr("rejection_reason",reason)
            raise OptunaTrialRejected(reason)
        value=float(row["mean_auc"])
        if not math.isfinite(value) or not 0<=value<=1:
            raise RuntimeError("Invalid objective value; study stopped.")
        trial.set_user_attr("repeat_sd",float(row["std_auc"]))
        trial.set_user_attr("mean_difference_vs_e4_leader",float(row["mean_difference_vs_e4_leader"]))
        trial.set_user_attr("wins_vs_e4_leader",int(row["wins_vs_e4_leader"]))
        trial.set_user_attr("losses_vs_e4_leader",int(row["losses_vs_e4_leader"]))
        return value

    def _export(self,study,config):
        study.trials_dataframe().to_csv(config.output_dir/"optuna_trials.csv",index=False)
        comparisons,repeats=[] , []
        for trial in study.trials:
            dest=config.output_dir/"trials"/f"trial_{trial.number:04d}"
            for filename,collection in (("e5b_comparison.csv",comparisons),
                                        ("e5b_paired_repeat_scores.csv",repeats)):
                path=dest/filename
                if path.is_file():
                    frame=pd.read_csv(path)
                    frame.insert(0,"trial_number",trial.number)
                    frame.insert(1,"trial_state",trial.state.name)
                    frame.insert(2,"optuna_value",trial.value)
                    collection.append(frame)
        for collection,filename in ((comparisons,"optuna_comparison.csv"),
                                     (repeats,"optuna_paired_repeat_scores.csv")):
            if collection:
                pd.concat(collection,ignore_index=True).to_csv(config.output_dir/filename,index=False)
            else:
                pd.DataFrame(columns=["trial_number","trial_state","optuna_value"]).to_csv(config.output_dir/filename,index=False)

    def _select(self,study,config):
        complete=[t for t in study.trials if t.state.name=="COMPLETE" and t.value is not None and math.isfinite(t.value)]
        if not complete:
            self._save_json(config.output_dir/"optuna_decision.json",dict(
                status="no_eligible_completed_trial",promote_over_e4=False,selected_trial_number=None))
            return False
        best=max(t.value for t in complete)
        tied=[t for t in complete if abs(t.value-best)<=1e-12]
        selected=min(tied,key=lambda t:(t.params["max_depth"],t.params["n_estimators"],t.number))
        dest=config.output_dir/"trials"/f"trial_{selected.number:04d}"
        row=pd.read_csv(dest/"e5b_comparison.csv").iloc[0]
        self._save_json(config.output_dir/"best_parameters.json",dict(
            trial_number=selected.number,mean_auc=selected.value,parameters=selected.params,
            artifact_directory=str(dest)))
        self._save_json(config.output_dir/"optuna_decision.json",dict(
            status="adaptive_development_selection_only",selected_trial_number=selected.number,
            optuna_best_trial_number=study.best_trial.number,mean_auc=selected.value,
            mean_difference_vs_e4_leader=float(row["mean_difference_vs_e4_leader"]),
            mean_difference_vs_e4_sparse=float(row["mean_difference_vs_e4_sparse"]),
            wins_vs_e4_leader=int(row["wins_vs_e4_leader"]),losses_vs_e4_leader=int(row["losses_vs_e4_leader"]),
            promote_over_e4=bool(float(row["mean_difference_vs_e4_leader"])>1e-12),
            tie_rule="Within 1e-12: lower depth, fewer estimators, lower trial number.",
            note="Best tuned CV score is not independent confirmation; no final refit/submission performed."))
        return True

    def run(self,config,dry_run=False,confirm_reference_alignment=False):
        context=self._prepare(config,confirm_reference_alignment)
        config.output_dir.mkdir(parents=True,exist_ok=False)
        (config.output_dir/"trials").mkdir()
        self._save_json(config.output_dir/"config.json",config.to_dict())
        self._save_json(config.output_dir/"base_config.json",context["base"].to_dict())
        context["references"].to_csv(config.output_dir/"reference_repeat_scores.csv",index=False)
        self._save_json(config.output_dir/"verified_split_indices.json",dict(
            index_type="zero_based_row_position",n_rows=context["base"].expected_rows,n_splits=5,
            split_seeds=list(range(10)),feature_names=context["features"],splits=context["splits"]))
        manifest=dict(status="validated",dry_run=bool(dry_run),started_utc=datetime.now(timezone.utc).isoformat(),
            n_trial_attempts_budget=config.n_trials,max_planned_fits=50*config.n_trials,
            sampler="TPESampler",sampler_seed=config.sampler_seed,n_startup_trials=config.n_startup_trials,
            optuna_n_jobs=1,xgboost_n_jobs=1,pruner="NopPruner",early_stopping=False,
            metric="mean_fold_auc_per_repeat",folds_per_trial=50,
            queued_reference_trials=config.reference_parameters(),queued_trials_count_toward_budget=True,
            input_and_source_sha256=context["hashes"],split_signature=context["split_signature"],
            versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                sklearn=sklearn.__version__,xgboost=context["xgboost_version"],optuna=context["optuna_version"]),
            model_seed=context["base"].model_seed,
            default_estimator_parameters=self.factory_type().parameters(context["base"],2,100),
            alignment="Saved split assignments verified; historical data contents/order and metric user-attested.",
            selection_note="Adaptive development tuning; no independent final evaluation.")
        path=config.output_dir/"run_manifest.json"
        self._save_json(path,manifest)
        if dry_run:
            self.logger.info("Optuna preflight passed. No study/trials/fits created; use a new full-run output directory.")
            return 0
        study=None
        try:
            optuna=self.optuna_module
            storage=f"sqlite:///{(config.output_dir/'study.sqlite3').as_posix()}"
            study=optuna.create_study(study_name=config.experiment_id,storage=storage,
                direction="maximize",load_if_exists=False,
                sampler=optuna.samplers.TPESampler(seed=config.sampler_seed,n_startup_trials=config.n_startup_trials),
                pruner=optuna.pruners.NopPruner())
            study.set_user_attr("protocol",config.to_dict())
            study.set_user_attr("split_signature",manifest["split_signature"])
            study.set_user_attr("historical_alignment_user_attested",True)
            for params in config.reference_parameters():
                study.enqueue_trial(params)
            manifest["status"]="running"
            self._save_json(path,manifest)
            def after_trial(current,trial):
                self._export(current,config)
                finished=sum(t.state.name in ("COMPLETE","FAIL","PRUNED") for t in current.trials)
                manifest.update(finished_trial_attempts=finished,
                    completed_trials=sum(t.state.name=="COMPLETE" for t in current.trials),
                    failed_trials=sum(t.state.name=="FAIL" for t in current.trials))
                self._save_json(path,manifest)
            study.optimize(lambda trial:self._objective(trial,config,context),n_trials=config.n_trials,
                n_jobs=1,catch=(OptunaTrialRejected,),callbacks=[after_trial],gc_after_trial=True)
            self._check_unchanged(context)
            self._export(study,config)
            selected=self._select(study,config)
            finished=sum(t.state.name in ("COMPLETE","FAIL","PRUNED") for t in study.trials)
            if finished!=config.n_trials:
                raise RuntimeError("Study did not consume the frozen trial-attempt budget.")
            manifest.update(status="completed",finished_trial_attempts=finished,
                completed_trials=sum(t.state.name=="COMPLETE" for t in study.trials),
                failed_trials=sum(t.state.name=="FAIL" for t in study.trials))
            self.logger.info("Optuna finished: %s",config.output_dir/"optuna_comparison.csv")
            return 0 if selected and manifest["failed_trials"]==0 else 2
        except BaseException as exc:
            manifest.update(status="interrupted_or_failed",error=f"{type(exc).__name__}: {exc}")
            self._save_json(config.output_dir/"study_error.json",dict(error=manifest["error"],traceback=traceback.format_exc()))
            if study is not None:
                try:
                    self._export(study,config)
                except Exception as export_error:
                    manifest["export_error"]=str(export_error)
            raise
        finally:
            manifest["finished_utc"]=datetime.now(timezone.utc).isoformat()
            self._save_json(path,manifest)

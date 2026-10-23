"""Bounded LightGBM tuning of ONE predeclared ensemble objective."""
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from src.experiments.lgbm_trial_rejected import LGBMTrialRejected


class LGBMOptunaExperiment:
    SHAPES = ["2:3", "3:3", "3:7", "4:3", "4:7", "4:15"]
    FIELDS = {"registry_config", "reference_run", "output_dir", "study_name", "n_trials", "sampler_seed", "objective"}

    def __init__(self, runner_type=None, optuna_module=None):
        self.runner_type = runner_type
        self.optuna_module = optuna_module

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def validate(cls, cfg):
        if not isinstance(cfg, dict) or set(cfg) != cls.FIELDS:
            raise ValueError(f"Expected exactly: {sorted(cls.FIELDS)}")
        if type(cfg["n_trials"]) is not int or not 1 <= cfg["n_trials"] <= 20:
            raise ValueError("n_trials must be an integer in [1,20], including the control trial.")
        if type(cfg["sampler_seed"]) is not int or not 0 <= cfg["sampler_seed"] < 2**32:
            raise ValueError("Invalid sampler seed.")
        if cfg["objective"] != "l1_xgb_lgbm":
            raise ValueError("This study predeclares l1_xgb_lgbm as its ONLY objective.")
        for name in cls.FIELDS - {"n_trials", "sampler_seed"}:
            if not isinstance(cfg[name], str) or not cfg[name].strip():
                raise ValueError(f"Invalid {name}.")
        return cfg

    @classmethod
    def control_suggestions(cls, parameters):
        shape = f"{parameters['max_depth']}:{parameters['num_leaves']}"
        values = dict(tree_shape=shape, min_child_samples=parameters["min_child_samples"],
            learning_rate=parameters["learning_rate"], n_estimators=parameters["n_estimators"],
            reg_lambda=parameters["reg_lambda"], colsample_bytree=parameters["colsample_bytree"])
        if (shape not in cls.SHAPES or not 5 <= values["min_child_samples"] <= 30
            or not .02 <= values["learning_rate"] <= .15 or values["n_estimators"] not in [50,100,200]
            or not .1 <= values["reg_lambda"] <= 30 or not .6 <= values["colsample_bytree"] <= 1.):
            raise ValueError("The fixed reference parameters are outside this predeclared search space.")
        return values

    @classmethod
    def suggest(cls, trial, fixed):
        shape = trial.suggest_categorical("tree_shape", cls.SHAPES)
        depth, leaves = map(int, shape.split(":"))
        parameters = dict(fixed, max_depth=depth, num_leaves=leaves,
            min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
            learning_rate=trial.suggest_float("learning_rate", .02, .15, log=True),
            n_estimators=trial.suggest_categorical("n_estimators", [50,100,200]),
            reg_lambda=trial.suggest_float("reg_lambda", .1, 30., log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", .6, 1.))
        return parameters

    def _runner(self):
        if self.runner_type is None:
            from src.experiments.registered_ensembles import RegisteredEnsembles
            self.runner_type = RegisteredEnsembles
        return self.runner_type()

    def _prepare(self, config_path, output=None):
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        helper = FrozenXGBoostSubmission()
        path = helper._path(config_path)
        cfg = self.validate(self.read(path))
        destination = helper._path(output or cfg["output_dir"])
        if destination.exists():
            raise FileExistsError("Output exists. Preserve it and choose a NEW --output directory; no in-place resume.")
        runner = self._runner()
        c = runner._prepare(cfg["registry_config"], destination, False)
        if c["cache"] is None or len(c["splits"]) != 50:
            raise ValueError("This bounded search requires verified cached L1/XGBoost predictions on all 50 folds.")
        base = c["cfg"]
        if set(base["models"]) != {"l1","xgboost","lightgbm"} or base["models"]["lightgbm"]["type"] != "lgbm_classifier":
            raise ValueError("Expected exactly the established l1, xgboost and lightgbm branches.")
        objective = base["ensembles"].get(cfg["objective"], {})
        if objective.get("members") != ["l1","xgboost","lightgbm"] or objective.get("weights") != [1,1,1]:
            raise ValueError("Keep the three-model objective in the exact order and weights [1,1,1].")
        self.control_suggestions(base["models"]["lightgbm"]["parameters"])
        reference = helper._path(cfg["reference_run"])
        manifest = self.read(reference/"run_manifest.json")
        if manifest.get("status") != "completed" or manifest.get("completed_folds") != 50 or manifest.get("control_auc_reproduced") is not True:
            raise ValueError("The fixed-LightGBM reference must be a completed, control-reproduced registry run.")
        required = {"settings.json","fold_scores.csv","comparison.csv","decision.json"}
        if not required <= manifest.get("artifact_sha256", {}).keys():
            raise ValueError("Reference artifact checksums incomplete.")
        extra = [reference/"run_manifest.json"]
        for name, digest in manifest["artifact_sha256"].items():
            artifact = (reference/name).resolve()
            if not artifact.is_relative_to(reference) or helper._sha(artifact) != digest:
                raise ValueError(f"Reference artifact changed: {name}")
            extra.append(artifact)
        settings = self.read(reference/"settings.json")
        if settings["config"] != base or settings["input_and_code_sha256"] != c["hashes"] or settings["versions"] != c["versions"]:
            raise ValueError("Fixed-reference inputs/code/config/environment differ. Do not bypass this check.")
        expected = pd.read_csv(reference/"fold_scores.csv").set_index(["repeat","fold"], verify_integrity=True)
        names = [*base["models"], *base["ensembles"]]
        if len(expected) != 50 or set(expected.index) != {(s["repeat"],s["fold"]) for s in c["splits"]} or not np.isfinite(expected[names]).all().all():
            raise ValueError("Invalid reference fold-score table.")
        if self.optuna_module is None:
            import optuna
            self.optuna_module = optuna
        versions = dict(c["versions"], optuna=self.optuna_module.__version__)
        extra.extend([path,Path(__file__),Path(inspect.getfile(LGBMTrialRejected)),
                      helper._path("scripts/run_lgbm_optuna.py"),helper._path("clil.py")])
        hashes = dict(c["hashes"])
        hashes.update({str(p.resolve()):helper._sha(p) for p in extra})
        return dict(config=cfg,config_path=path,output=destination,runner=runner,helper=helper,
            context=c,base=base,reference=reference,expected=expected,versions=versions,hashes=hashes)

    @staticmethod
    def _unchanged(c):
        for path,digest in c["hashes"].items():
            if c["helper"]._sha(path) != digest:
                raise RuntimeError(f"Study input/code changed: {path}")

    def _check_trial(self,c,run,number,parameters):
        h=c["helper"]
        manifest=self.read(run/"run_manifest.json")
        if manifest.get("status")!="completed" or manifest.get("completed_folds")!=50 or manifest.get("control_auc_reproduced") is not True:
            raise RuntimeError("Trial did not complete valid 50-fold evaluation.")
        if manifest.get("planned_fits")!=50 or manifest.get("fits_performed")!=50:
            raise RuntimeError("Trial did not reuse L1/XGBoost; fixed fit budget violated.")
        actual=pd.read_csv(run/"fold_scores.csv").set_index(["repeat","fold"],verify_integrity=True)
        expected=c["expected"]
        if set(actual.index)!=set(expected.index):raise RuntimeError("Trial fold keys changed.")
        actual=actual.loc[expected.index]
        controls=["l1","xgboost",c["base"]["baseline"]]
        if number==0:controls=[*c["base"]["models"],*c["base"]["ensembles"]]
        for name in controls:
            if not np.allclose(actual[name],expected[name],rtol=0,atol=1e-12):
                raise RuntimeError(f"Trial {number} failed historical {name} AUC reproduction. Do not relax tolerance.")
        row=pd.read_csv(run/"comparison.csv")
        selected=row.loc[row.candidate==c["config"]["objective"]]
        if len(selected)!=1:raise RuntimeError("Objective candidate missing/duplicated.")
        r=selected.iloc[0]
        value=float(actual[c["config"]["objective"]].groupby(level=0).mean().mean())
        if not np.isfinite(value) or not np.isclose(value,r.mean_auc,rtol=0,atol=1e-12):
            raise RuntimeError("Objective aggregate differs from registry comparison.")
        return dict(number=int(number),state="COMPLETE",objective_auc=value,
            difference_vs_baseline=float(r.difference_vs_baseline),wins=int(r.wins),losses=int(r.losses),
            promotable=bool(r.promotable),parameters=parameters,cv_run=str(run.resolve()),
            reference_reproduced=True,registry_artifacts=manifest["artifact_sha256"],
            registry_manifest_sha256=h._sha(run/"run_manifest.json"))

    def search(self,config_path,output=None,dry_run=False):
        c=self._prepare(config_path,output);h=c["helper"];cfg=c["config"];out=c["output"]
        if dry_run:
            return dict(status="preflight_passed",objective=cfg["objective"],n_trials=cfg["n_trials"],
                max_new_fits=50*cfg["n_trials"],cached_models=["l1","xgboost"],control_trial=0,
                early_stopping=False,pruning=False,test_opened=False,output=str(out))
        out.mkdir(parents=True,exist_ok=False);(out/"trials").mkdir()
        h._write(out/"settings.json",dict(config=cfg,config_path=str(c["config_path"]),versions=c["versions"],
            input_and_code_sha256=c["hashes"],base_registry_configuration=c["base"],
            objective="Mean fold AUC per repeat, then mean across 10 repeats, for fixed equal-weight l1_xgb_lgbm only.",
            search_space=dict(tree_shape=self.SHAPES,min_child_samples=[5,30],learning_rate_log=[.02,.15],
                n_estimators=[50,100,200],reg_lambda_log=[.1,30],colsample_bytree=[.6,1.]),
            policy="All other parameters/weights fixed. No early stopping, pruning, public-score selection or in-place resume."))
        manifest=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),planned_trials=cfg["n_trials"],
            max_new_fits=50*cfg["n_trials"],completed_trials=0,failed_trials=0,test_opened=False,kaggle_upload_performed=False)
        h._write(out/"run_manifest.json",manifest)
        records=[];study=None
        try:
            api=self.optuna_module
            study=api.create_study(study_name=cfg["study_name"],storage=f"sqlite:///{(out/'study.sqlite3').as_posix()}",
                direction="maximize",sampler=api.samplers.TPESampler(seed=cfg["sampler_seed"],n_startup_trials=5),
                pruner=api.pruners.NopPruner(),load_if_exists=False)
            fixed=c["base"]["models"]["lightgbm"]["parameters"]
            study.enqueue_trial(self.control_suggestions(fixed))
            def objective(trial):
                self._unchanged(c)
                root=out/"trials"/f"trial_{trial.number:04d}";root.mkdir(parents=True,exist_ok=False)
                parameters=self.suggest(trial,fixed)
                if trial.number==0 and parameters!=fixed:
                    raise RuntimeError("Control trial parameters differ from original LightGBM constructor.")
                trial_cfg=deepcopy(c["base"])
                trial_cfg["models"]["lightgbm"]["parameters"]=parameters
                run=root/"cv";trial_cfg["output_dir"]=str(run.resolve())
                path=root/"registry_config.json";h._write(path,trial_cfg)
                trial.set_user_attr("resolved_lightgbm_parameters",parameters)
                trial.set_user_attr("cv_run",str(run.resolve()))
                try:
                    self._runner().cv(str(path.resolve()),output=str(run.resolve()),no_cache=False,dry_run=False)
                    record=self._check_trial(c,run,trial.number,parameters)
                except RuntimeError as exc:
                    if trial.number==0 or not str(exc).startswith("Python warnings require review"):
                        raise
                    record=dict(number=trial.number,state="FAIL",reason=str(exc),cv_run=str(run.resolve()),parameters=parameters)
                    records.append(record);h._write(root/"trial_result.json",record)
                    h._write(out/"trial_results.json",records)
                    manifest["failed_trials"]+=1;h._write(out/"run_manifest.json",manifest)
                    raise LGBMTrialRejected(str(exc)) from exc
                records.append(record);h._write(root/"trial_result.json",record)
                h._write(out/"trial_results.json",records)
                manifest["completed_trials"]+=1
                h._write(out/"run_manifest.json",manifest)
                self._unchanged(c)
                trial.set_user_attr("promotable",record["promotable"])
                return record["objective_auc"]
            study.optimize(objective,n_trials=cfg["n_trials"],n_jobs=1,catch=(LGBMTrialRejected,),gc_after_trial=True)
            self._unchanged(c)
            complete=[r for r in records if r["state"]=="COMPLETE"]
            if not complete or len(records)!=cfg["n_trials"]:
                raise RuntimeError("Trial accounting incomplete.")
            best=min(complete,key=lambda r:(-r["objective_auc"],r["number"]))
            h._write(out/"best_trial.json",best)
            selected=best if best["promotable"] else None
            decision=dict(objective=cfg["objective"],best_trial=best["number"],best_objective_auc=best["objective_auc"],
                selected_trial=None if selected is None else best["number"],promote=selected is not None,
                rule="Choose the highest objective-AUC trial (ties: earliest). Promote only if it beats existing L1+XGBoost on mean AUC and >=6/10 repeats. No fallback trial selection.",
                interpretation="Adaptive development tuning; not nested CV, independent evaluation or a guaranteed leaderboard improvement.")
            h._write(out/"decision.json",decision)
            pd.DataFrame([{k:v for k,v in r.items() if k not in ("parameters","registry_artifacts")} for r in records]).to_csv(out/"trials.csv",index=False)
            manifest.update(status="completed",control_trial_reproduced=True,promote=decision["promote"],best_trial=best["number"])
            manifest["artifact_sha256"]={str(p.relative_to(out)):h._sha(p) for p in out.rglob("*")
                if p.is_file() and p.name!="run_manifest.json" and not p.name.startswith("study.sqlite3")}
            # Include trial manifests explicitly (root manifest alone is excluded).
            for p in (out/"trials").rglob("run_manifest.json"):
                manifest["artifact_sha256"][str(p.relative_to(out))]=h._sha(p)
            return decision
        except BaseException as exc:
            manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
            h._write(out/"error.json",dict(error=manifest["error"],traceback=traceback.format_exc()));raise
        finally:
            manifest["finished_utc"]=datetime.now(timezone.utc).isoformat();h._write(out/"run_manifest.json",manifest)

    def predict(self,run_path,output,candidate="auto",test=None,template=None,dry_run=False):
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        h=FrozenXGBoostSubmission();run=h._path(run_path)
        manifest=self.read(run/"run_manifest.json")
        if manifest.get("status")!="completed" or manifest.get("control_trial_reproduced") is not True:
            raise ValueError("Need a completed, control-reproduced Optuna study.")
        for name,digest in manifest["artifact_sha256"].items():
            path=(run/name).resolve()
            if not path.is_relative_to(run) or h._sha(path)!=digest:
                raise ValueError(f"Study artifact changed: {name}")
        decision=self.read(run/"decision.json");best=self.read(run/"best_trial.json")
        if not decision.get("promote") or decision.get("selected_trial")!=best["number"]:
            raise ValueError("Best objective trial did not pass promotion. Retain existing 0.91514 submission.")
        settings=self.read(run/"settings.json")
        c=self._prepare(settings["config_path"],output)
        self._unchanged(c)
        if c["hashes"]!=settings["input_and_code_sha256"] or c["versions"]!=settings["versions"] or c["config"]!=settings["config"]:
            raise ValueError("Study inputs/config/code/environment changed.")
        if candidate not in ("auto",c["config"]["objective"]):
            raise ValueError("Only the predeclared objective candidate can be exported from this study.")
        target=h._path(best["cv_run"])
        if not target.is_relative_to(run/"trials"):
            raise ValueError("Best trial path outside study.")
        return self._runner().predict(str(target),output,c["config"]["objective"],test,template,dry_run)

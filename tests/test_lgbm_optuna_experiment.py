"""Synthetic API orchestration tests; native libraries and upstream adapters are optional."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments.lgbm_optuna_experiment import LGBMOptunaExperiment


class Helper:
    root=None
    @classmethod
    def _path(cls,path):return (cls.root/Path(path)).resolve()
    @staticmethod
    def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @staticmethod
    def _write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False),encoding="utf-8")


class Trial:
    def __init__(self,number,fixed=None):self.number=number;self.fixed=fixed or {};self.params={};self.user_attrs={}
    def suggest_categorical(self,name,choices):
        value=self.fixed.get(name,choices[-1]);self.params[name]=value;return value
    def suggest_int(self,name,low,high):
        value=self.fixed.get(name,low);self.params[name]=value;return value
    def suggest_float(self,name,low,high,log=False):
        value=self.fixed.get(name,(low+high)/2);self.params[name]=value;return value
    def set_user_attr(self,name,value):self.user_attrs[name]=value


class Study:
    def __init__(self):self.queue=[];self.trials=[]
    def enqueue_trial(self,value):self.queue.append(value)
    def optimize(self,objective,n_trials,n_jobs,catch,gc_after_trial):
        if n_jobs!=1:raise AssertionError("Must run sequentially")
        for i in range(n_trials):
            trial=Trial(i,self.queue.pop(0) if self.queue else None)
            try:trial.value=objective(trial);trial.state="COMPLETE"
            except catch:trial.value=None;trial.state="FAIL"
            self.trials.append(trial)


class FakeOptuna:
    __version__="TEST_DOUBLE_NOT_NATIVE_OPTUNA"
    samplers=types.SimpleNamespace(TPESampler=lambda **kwargs:kwargs)
    pruners=types.SimpleNamespace(NopPruner=lambda:None)
    def create_study(self,**kwargs):
        self.options=kwargs;self.study=Study()
        Path(kwargs["storage"].removeprefix("sqlite:///")).write_text("TEST DOUBLE, not a SQLite database")
        return self.study


class Registry:
    context=None;mode="ok";calls=[]
    def _prepare(self,path,output,no_cache):
        if Path(output).exists():raise FileExistsError("Output exists")
        return dict(type(self).context,output=Path(output))
    def cv(self,config,output,no_cache,dry_run):
        cfg=json.loads(Path(config).read_text());out=Path(output);out.mkdir()
        number=int(out.parent.name.split("_")[-1]);type(self).calls.append(number)
        if (self.mode=="warning" and number==1) or (self.mode=="warning-control" and number==0):raise RuntimeError("Python warnings require review; inspect warnings.json and fit.log.")
        frame=type(self).context["expected_fixture"].copy()
        if number:
            frame["lightgbm"]+=.02;frame["l1_xgb_lgbm"]+=.01
            frame["lgbm_l1"]=.95  # Deliberately better secondary score; NOT the objective.
        if self.mode=="bad-control" and number==0:frame["lightgbm"]+=.1
        if self.mode=="no-gain":frame["l1_xgb_lgbm"]=.90
        frame.to_csv(out/"fold_scores.csv",index=False)
        names=[*cfg["models"],*cfg["ensembles"]];rows=[]
        for name in names:
            value=float(frame[name].mean());difference=value-float(frame[cfg["baseline"]].mean())
            rows.append(dict(candidate=name,mean_auc=value,difference_vs_baseline=difference,
                wins=10 if difference>1e-12 else 0,losses=10 if difference< -1e-12 else 0,
                promotable=name in cfg["ensembles"] and name!=cfg["baseline"] and difference>1e-12))
        pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
        Helper._write(out/"decision.json",dict(selected="lgbm_l1" if number else None))
        artifacts={p.name:Helper._sha(p) for p in out.iterdir()}
        Helper._write(out/"run_manifest.json",dict(status="completed",completed_folds=50,control_auc_reproduced=True,
            planned_fits=50,fits_performed=50,artifact_sha256=artifacts))
    def predict(self,run,output,candidate,test,template,dry_run):
        type(self).last_prediction=dict(run=run,output=output,candidate=candidate,dry_run=dry_run)
        return dict(status="prediction_preflight_passed" if dry_run else "ready_for_manual_submission",candidate=candidate)


class LGBMOptunaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();Helper.root=self.root
        (self.root/"scripts").mkdir();(self.root/"scripts/run_lgbm_optuna.py").write_text("# source fixture\n")
        (self.root/"clil.py").write_text("# source fixture\n")
        self.fixed=dict(objective="binary",learning_rate=.05,n_estimators=100,max_depth=3,num_leaves=7,
            min_child_samples=20,reg_lambda=10.,subsample=1.,colsample_bytree=1.,random_state=42,n_jobs=1,
            verbosity=1,deterministic=True,force_col_wise=True)
        self.base=dict(frozen_dir="frozen",l1_config="l1",reuse_cv_run="old",output_dir="unused",
            models=dict(l1=dict(type="l1_pipeline"),xgboost=dict(type="xgb_topk"),lightgbm=dict(type="lgbm_classifier",parameters=self.fixed)),
            ensembles=dict(l1_xgb=dict(members=["l1","xgboost"],weights=[1,1]),
                lgbm_l1=dict(members=["lightgbm","l1"],weights=[1,1]),
                lgbm_xgb=dict(members=["lightgbm","xgboost"],weights=[1,1]),
                l1_xgb_lgbm=dict(members=["l1","xgboost","lightgbm"],weights=[1,1,1])),baseline="l1_xgb")
        self.registry_config=self.root/"registry.json";Helper._write(self.registry_config,self.base)
        sentinel=self.root/"labelled_input";sentinel.write_text("UNCHANGED")
        hashes={str(sentinel):Helper._sha(sentinel),str(self.registry_config):Helper._sha(self.registry_config)}
        versions=dict(python="fixture",lightgbm="fixture",xgboost="fixture")
        expected=pd.DataFrame([dict(repeat=r,split_seed=r,fold=f,l1=.862112,xgboost=.890672,
            lightgbm=.856224,l1_xgb=.902878,lgbm_l1=.884225,lgbm_xgb=.886114,l1_xgb_lgbm=.900614)
            for r in range(10) for f in range(5)])
        Registry.context=dict(cache=object(),splits=[dict(repeat=r,fold=f) for r in range(10) for f in range(5)],
            cfg=self.base,hashes=hashes,versions=versions,expected_fixture=expected)
        Registry.mode="ok";Registry.calls=[]
        reference=self.root/"reference";reference.mkdir()
        Helper._write(reference/"settings.json",dict(config=self.base,input_and_code_sha256=hashes,versions=versions))
        expected.to_csv(reference/"fold_scores.csv",index=False)
        (reference/"comparison.csv").write_text("candidate,mean_auc\nl1_xgb_lgbm,0.900614\n")
        Helper._write(reference/"decision.json",dict(selected=None))
        Helper._write(reference/"run_manifest.json",dict(status="completed",completed_folds=50,control_auc_reproduced=True,
            artifact_sha256={p.name:Helper._sha(p) for p in reference.iterdir()}))
        self.config=dict(registry_config=str(self.registry_config),reference_run=str(reference),output_dir=str(self.root/"study"),
            study_name="test",n_trials=3,sampler_seed=42,objective="l1_xgb_lgbm")
        self.path=self.root/"optuna.json";Helper._write(self.path,self.config)
        module=types.ModuleType("src.submission.frozen_xgboost_submission");module.FrozenXGBoostSubmission=Helper
        self.patch=patch.dict("sys.modules",{"src.submission.frozen_xgboost_submission":module});self.patch.start()
        self.api=FakeOptuna();self.runner=LGBMOptunaExperiment(Registry,self.api)
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_config_valid(self):self.assertEqual(self.runner.validate(self.config)["n_trials"],3)
    def test_trial_budget_rejected(self):
        with self.assertRaises(ValueError):self.runner.validate(dict(self.config,n_trials=21))
    def test_boolean_trial_budget_rejected(self):
        with self.assertRaises(ValueError):self.runner.validate(dict(self.config,n_trials=True))
    def test_objective_cannot_switch(self):
        with self.assertRaises(ValueError):self.runner.validate(dict(self.config,objective="lightgbm"))
    def test_unknown_settings_rejected(self):
        with self.assertRaises(ValueError):self.runner.validate(dict(self.config,weights=[1,1,2]))
    def test_control_parameter_roundtrip(self):
        trial=Trial(0,self.runner.control_suggestions(self.fixed))
        self.assertEqual(self.runner.suggest(trial,self.fixed),self.fixed)
    def test_all_shapes_obey_depth_limit(self):
        for shape in self.runner.SHAPES:
            depth,leaves=map(int,shape.split(":"));self.assertLessEqual(leaves,2**depth)
    def test_out_of_space_reference_rejected(self):
        with self.assertRaises(ValueError):self.runner.control_suggestions(dict(self.fixed,num_leaves=31))
    def test_preflight_no_fit_no_output(self):
        result=self.runner.search(self.path,dry_run=True)
        self.assertEqual(result["max_new_fits"],150);self.assertFalse((self.root/"study").exists());self.assertFalse(Registry.calls)
    def test_missing_cache_rejected(self):
        Registry.context["cache"]=None
        with self.assertRaises(ValueError):self.runner.search(self.path,dry_run=True)
    def test_changed_reference_rejected(self):
        (self.root/"reference/comparison.csv").write_text("changed")
        with self.assertRaises(ValueError):self.runner.search(self.path,dry_run=True)
    def test_complete_search_objective_not_secondary_winner(self):
        result=self.runner.search(self.path)
        self.assertTrue(result["promote"]);self.assertEqual(result["best_trial"],1)
        best=self.runner.read(self.root/"study/best_trial.json")
        self.assertAlmostEqual(best["objective_auc"],.910614)
        self.assertEqual(len(self.api.study.trials),3)
        self.assertEqual(self.api.options["direction"],"maximize")
        self.assertIs(self.api.options["load_if_exists"],False)
        self.assertEqual(Registry.calls,[0,1,2])
    def test_failed_control_aborts(self):
        Registry.mode="bad-control"
        with self.assertRaises(RuntimeError):self.runner.search(self.path)
        manifest=self.runner.read(self.root/"study/run_manifest.json")
        self.assertEqual(manifest["status"],"failed_or_interrupted");self.assertEqual(Registry.calls,[0])
    def test_warning_trial_rejected_but_budget_includes_failure(self):
        Registry.mode="warning";self.runner.search(self.path)
        manifest=self.runner.read(self.root/"study/run_manifest.json")
        self.assertEqual(manifest["failed_trials"],1);self.assertEqual(manifest["completed_trials"],2)
        self.assertEqual(len(self.api.study.trials),3)
    def test_no_in_place_resume(self):
        (self.root/"study").mkdir()
        with self.assertRaises(FileExistsError):self.runner.search(self.path,dry_run=True)
    def test_export_delegates_exact_objective(self):
        self.runner.search(self.path)
        result=self.runner.predict(self.root/"study",self.root/"prediction",dry_run=True)
        self.assertEqual(result["candidate"],"l1_xgb_lgbm")
        self.assertTrue(Registry.last_prediction["run"].endswith("trial_0001/cv"))
    def test_secondary_candidate_export_rejected(self):
        self.runner.search(self.path)
        with self.assertRaises(ValueError):self.runner.predict(self.root/"study",self.root/"prediction",candidate="lgbm_l1")
    def test_changed_study_artifact_rejected(self):
        self.runner.search(self.path);(self.root/"study/best_trial.json").write_text("{}")
        with self.assertRaises(ValueError):self.runner.predict(self.root/"study",self.root/"prediction")
    def test_changed_source_rejected_before_export(self):
        self.runner.search(self.path);(self.root/"labelled_input").write_text("CHANGED")
        with self.assertRaises((ValueError,RuntimeError)):self.runner.predict(self.root/"study",self.root/"prediction")
    def test_no_promotion_no_export(self):
        # Keep trial0 reproducible; all suggested trials become worse than baseline.
        self.config["n_trials"]=1;Helper._write(self.path,self.config)
        result=self.runner.search(self.path);self.assertFalse(result["promote"])
        with self.assertRaises(ValueError):self.runner.predict(self.root/"study",self.root/"prediction")

    def test_full_twenty_trial_budget_includes_control(self):
        self.config["n_trials"]=20;Helper._write(self.path,self.config)
        self.runner.search(self.path)
        manifest=self.runner.read(self.root/"study/run_manifest.json")
        self.assertEqual(len(self.api.study.trials),20)
        self.assertEqual(Registry.calls,list(range(20)))
        self.assertEqual(manifest["max_new_fits"],1000)
    def test_control_warning_aborts_instead_of_rejecting(self):
        Registry.mode="warning-control"
        with self.assertRaises(RuntimeError):self.runner.search(self.path)
        self.assertEqual(Registry.calls,[0])
        self.assertEqual(self.runner.read(self.root/"study/run_manifest.json")["status"],"failed_or_interrupted")
    @unittest.skipUnless(importlib.util.find_spec("optuna") is not None,"Native Optuna unavailable in build environment")
    def test_real_optuna_sqlite_control_and_budget(self):
        import optuna
        study=optuna.create_study(direction="maximize",storage=f"sqlite:///{(self.root/'real.sqlite3').as_posix()}",
            sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.NopPruner())
        study.enqueue_trial(self.runner.control_suggestions(self.fixed))
        values=[]
        def objective(trial):
            params=self.runner.suggest(trial,self.fixed);values.append(params);return float(params["num_leaves"])
        study.optimize(objective,n_trials=2,n_jobs=1)
        self.assertEqual(values[0],self.fixed);self.assertEqual(len(study.trials),2)
    @unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None,"Native LightGBM unavailable in build environment")
    def test_real_lightgbm_constructor_and_probabilities(self):
        from lightgbm import LGBMClassifier
        rng=np.random.default_rng(42);x=pd.DataFrame(rng.normal(size=(100,4)),columns=list("abcd"));y=np.tile([0,1],50)
        model=LGBMClassifier(**self.fixed).fit(x,y);p=model.predict_proba(x)
        self.assertEqual(p.shape,(100,2));self.assertTrue(np.isfinite(p).all())


if __name__=="__main__":unittest.main()

"""Tests for the Optuna adapter, separate from existing E5b data/CV tests."""
from dataclasses import dataclass,replace
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from src.config.xgboost_optuna_config import XGBoostOptunaConfig
from src.modeling.xgboost_pipeline_factory import XGBoostPipelineFactory
from src.experiments.xgboost_optuna_runner import XGBoostOptunaRunner
from src.experiments.optuna_trial_rejected import OptunaTrialRejected


@dataclass(frozen=True)
class FakeBaseConfig:
    experiment_id:str="fixture"
    output_dir:Path=Path("unused")
    max_depth_values:tuple=(1,2)
    n_estimators_values:tuple=(50,100,200)
    model_seed:int=42
    expected_rows:int=250
    def to_dict(self):
        return dict(experiment_id=self.experiment_id,output_dir=str(self.output_dir),
                    max_depth_values=list(self.max_depth_values),n_estimators_values=list(self.n_estimators_values),
                    model_seed=self.model_seed,expected_rows=self.expected_rows)


class FakeFactory(XGBoostPipelineFactory):
    @staticmethod
    def check_environment():
        return "TEST_DOUBLE_NOT_REAL_XGBOOST"


class FakeCVRunner:
    """Adapter test double: emits records; does NOT fit any classifier."""
    behavior="success"
    calls=[]
    def __init__(self,factory=None):
        self.factory=factory
    def run(self,config,confirm_reference_alignment=False):
        if self.behavior=="interrupt":
            raise KeyboardInterrupt
        if self.behavior=="infrastructure_error":
            raise OSError("injected disk problem")
        if not confirm_reference_alignment:
            raise AssertionError("Confirmation not passed through.")
        depth,trees=config.max_depth_values[0],config.n_estimators_values[0]
        parameters=self.factory.parameters(config,depth,trees)
        self.calls.append(parameters)
        config.output_dir.mkdir(parents=True,exist_ok=False)
        warned=self.behavior=="warning"
        failed=self.behavior=="invalid"
        score=.83+depth*.002+min(trees,500)*.00001
        pd.DataFrame([dict(experiment_id=config.experiment_id,completed_folds=50,
            valid_folds=49 if failed else 50,failed_folds=int(failed),warning_folds=int(warned),
            eligible=not(warned or failed),mean_auc=score,std_auc=.01,
            mean_difference_vs_e4_leader=score-.862111696676914,
            mean_difference_vs_e4_sparse=score-.8528636259071043,
            wins_vs_e4_leader=0,losses_vs_e4_leader=10)]).to_csv(config.output_dir/"e5b_comparison.csv",index=False)
        pd.DataFrame([dict(repeat=r,split_seed=r,n_folds=5,mean_auc=score) for r in range(10)]).to_csv(
            config.output_dir/"e5b_paired_repeat_scores.csv",index=False)
        (config.output_dir/"run_manifest.json").write_text(json.dumps(dict(status="completed",completed_fits=50)))
        return 2 if warned or failed else 0


class FakeTrial:
    def __init__(self,number,params=None):
        self.number=number
        self.fixed=dict(params or {})
        self.params={}
        self.user_attrs={}
        self.state=types.SimpleNamespace(name="RUNNING")
        self.value=None
    def set_user_attr(self,name,value):
        self.user_attrs[name]=value
    def suggest_int(self,name,low,high):
        value=self.fixed.get(name,low)
        self.params[name]=value
        return value
    def suggest_float(self,name,low,high,log=False):
        value=self.fixed.get(name,low)
        self.params[name]=value
        return value
    def suggest_categorical(self,name,choices):
        value=self.fixed.get(name,choices[0])
        self.params[name]=value
        return value


class FakeStudy:
    """No SQLite or TPE: validates calls to the optimization API only."""
    def __init__(self):
        self.trials=[]
        self.queued=[]
        self.user_attrs={}
    def set_user_attr(self,name,value):
        self.user_attrs[name]=value
    def enqueue_trial(self,params):
        self.queued.append(dict(params))
    def optimize(self,objective,n_trials,n_jobs,catch,callbacks,gc_after_trial):
        if n_jobs!=1:
            raise AssertionError("Parallel optimization is forbidden.")
        for i in range(n_trials):
            trial=FakeTrial(i,self.queued[i] if i<len(self.queued) else None)
            self.trials.append(trial)
            try:
                trial.value=objective(trial)
                trial.state=types.SimpleNamespace(name="COMPLETE")
            except catch:
                trial.state=types.SimpleNamespace(name="FAIL")
            except BaseException:
                trial.state=types.SimpleNamespace(name="FAIL")
                raise
            for callback in callbacks:
                callback(self,trial)
    @property
    def best_trial(self):
        return max((t for t in self.trials if t.state.name=="COMPLETE"),key=lambda t:t.value)
    def trials_dataframe(self):
        return pd.DataFrame([dict(number=t.number,state=t.state.name,value=t.value,**t.params) for t in self.trials])


class FakeOptuna:
    __version__="TEST_DOUBLE_NOT_REAL_OPTUNA"
    samplers=types.SimpleNamespace(TPESampler=lambda **kwargs:dict(kwargs))
    pruners=types.SimpleNamespace(NopPruner=lambda:"NOP")
    def __init__(self):
        self.created=[]
    def create_study(self,**kwargs):
        self.created.append(kwargs)
        self.study=FakeStudy()
        return self.study


class OptunaExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.config=XGBoostOptunaConfig(output_dir=self.root/"output",base_config_path=self.root/"base.json",
                                        n_trials=3,n_startup_trials=2)
        self.input=self.root/"immutable.txt"
        self.input.write_text("unchanged")
        self.engine=FakeOptuna()
        self.runner=XGBoostOptunaRunner(runner_type=FakeCVRunner,factory_type=FakeFactory,optuna_module=self.engine)
        self.context=dict(base=FakeBaseConfig(),splits=[],references=pd.DataFrame({"repeat":range(10)}),
            features=[str(i) for i in range(150)],
            hashes={str(self.input):self.runner._sha256(self.input)},split_signature="fixture_signature",
            xgboost_version="TEST_DOUBLE_NOT_REAL_XGBOOST",optuna_version="TEST_DOUBLE_NOT_REAL_OPTUNA")
        FakeCVRunner.behavior="success"
        FakeCVRunner.calls=[]
    def tearDown(self):
        self.temp.cleanup()
    def run_fake(self,**kwargs):
        with patch.object(self.runner,"_prepare",return_value=self.context):
            return self.runner.run(self.config,confirm_reference_alignment=True,**kwargs)
    def objective(self):
        return self.runner._objective(FakeTrial(0,self.config.reference_parameters()[0]),self.config,self.context)

    def test_default_budget_and_search_space(self):
        default=XGBoostOptunaConfig()
        self.assertEqual(default.n_trials,30)
        self.assertEqual(set(default.search_space),{"max_depth","n_estimators","learning_rate","min_child_weight","reg_lambda","subsample","colsample_bytree"})
    def test_config_relative_paths_normalized(self):
        p=self.root/"config.json"
        p.write_text(json.dumps({"base_config_path":"base.json","output_dir":"output"}))
        config=XGBoostOptunaConfig.from_json(p,project_root=self.root)
        self.assertEqual(config.base_config_path,self.root/"base.json")
        self.assertEqual(config.output_dir,self.root/"output")
    def test_unknown_config_field_rejected(self):
        p=self.root/"bad.json";p.write_text('{"n_jobs":8}')
        with self.assertRaises(ValueError):
            XGBoostOptunaConfig.from_json(p)
    def test_budget_includes_two_reference_trials(self):
        with self.assertRaises(ValueError):
            replace(self.config,n_trials=1)
    def test_invalid_ranges_and_unknown_parameter_rejected(self):
        space=json.loads(json.dumps(self.config.search_space))
        space["subsample"]["high"]=1.1
        with self.assertRaises(ValueError):replace(self.config,search_space=space)
        space=json.loads(json.dumps(self.config.search_space));space["random_state"]={"type":"int","low":1,"high":100}
        with self.assertRaises(ValueError):replace(self.config,search_space=space)
    def test_queued_reference_values_must_be_in_space(self):
        space=json.loads(json.dumps(self.config.search_space));space["max_depth"]["high"]=2
        with self.assertRaises(ValueError):replace(self.config,search_space=space)
    def test_original_factory_defaults_preserved(self):
        parameters=XGBoostPipelineFactory().parameters(FakeBaseConfig(),2,100)
        self.assertEqual(parameters["learning_rate"],.05)
        self.assertEqual(parameters["min_child_weight"],5)
        self.assertEqual(parameters["reg_lambda"],10.)
        self.assertEqual(parameters["random_state"],42)
        self.assertEqual(parameters["n_jobs"],1)
    def test_factory_overrides_are_instance_local(self):
        f=XGBoostPipelineFactory(overrides={"learning_rate":.1,"reg_lambda":1.})
        self.assertEqual(f.parameters(FakeBaseConfig(),2,100)["learning_rate"],.1)
        self.assertEqual(XGBoostPipelineFactory().parameters(FakeBaseConfig(),2,100)["learning_rate"],.05)
    def test_factory_cannot_override_seed_or_jobs(self):
        for key in ("random_state","n_jobs","n_estimators","max_depth","early_stopping_rounds"):
            with self.subTest(key=key),self.assertRaises(ValueError):XGBoostPipelineFactory({key:2})
    def test_bad_override_rejected(self):
        for value in (-1,float("nan"),True,2):
            with self.subTest(value=value),self.assertRaises(ValueError):XGBoostPipelineFactory({"subsample":value})
    def test_missing_confirmation_rejected(self):
        with self.assertRaises(ValueError):self.runner._prepare(self.config,False)
    def test_existing_output_rejected(self):
        self.config.output_dir.mkdir()
        with self.assertRaises(FileExistsError):self.runner._prepare(self.config,True)
    def test_source_mutation_stops_trial(self):
        self.input.write_text("changed")
        with self.assertRaises(RuntimeError):self.objective()
        self.assertFalse(FakeCVRunner.calls)
    def test_objective_passes_sampled_parameters(self):
        value=self.objective()
        self.assertAlmostEqual(value,.835)
        p=FakeCVRunner.calls[0]
        self.assertEqual(p["max_depth"],2)
        self.assertEqual(p["n_estimators"],100)
        self.assertEqual(p["learning_rate"],.05)
        self.assertEqual(p["random_state"],42)
    def test_warning_trial_rejected_not_assigned_zero(self):
        FakeCVRunner.behavior="warning"
        with self.assertRaises(OptunaTrialRejected):self.objective()
        self.assertTrue((self.config.output_dir/"trials/trial_0000/e5b_comparison.csv").exists())
    def test_failed_fold_trial_rejected(self):
        FakeCVRunner.behavior="invalid"
        with self.assertRaises(OptunaTrialRejected):self.objective()
    def test_unexpected_error_not_swallowed(self):
        FakeCVRunner.behavior="infrastructure_error"
        with self.assertRaises(OSError):self.objective()
    def test_preflight_creates_no_study(self):
        self.assertEqual(self.run_fake(dry_run=True),0)
        self.assertFalse(self.engine.created)
        self.assertFalse((self.config.output_dir/"study.sqlite3").exists())
    def test_full_adapter_orchestration_with_test_doubles(self):
        self.assertEqual(self.run_fake(),0)
        self.assertEqual(len(FakeCVRunner.calls),3)
        self.assertEqual(len(self.engine.study.trials),3)
        self.assertEqual(len(self.engine.study.queued),2)
        self.assertEqual(self.engine.created[0]["direction"],"maximize")
        self.assertFalse(self.engine.created[0]["load_if_exists"])
        self.assertEqual(self.engine.created[0]["sampler"]["seed"],42)
        table=pd.read_csv(self.config.output_dir/"optuna_comparison.csv")
        self.assertEqual(len(table),3)
        decision=json.loads((self.config.output_dir/"optuna_decision.json").read_text())
        self.assertEqual(decision["selected_trial_number"],1)
        manifest=json.loads((self.config.output_dir/"run_manifest.json").read_text())
        self.assertEqual(manifest["status"],"completed")
        self.assertEqual(manifest["finished_trial_attempts"],3)
    def test_all_rejected_study_is_preserved_without_best(self):
        FakeCVRunner.behavior="warning"
        self.assertEqual(self.run_fake(),2)
        self.assertTrue(all(t.state.name=="FAIL" for t in self.engine.study.trials))
        self.assertFalse((self.config.output_dir/"best_parameters.json").exists())
        decision=json.loads((self.config.output_dir/"optuna_decision.json").read_text())
        self.assertIsNone(decision["selected_trial_number"])
    def test_interruption_marks_manifest(self):
        FakeCVRunner.behavior="interrupt"
        with self.assertRaises(KeyboardInterrupt):self.run_fake()
        manifest=json.loads((self.config.output_dir/"run_manifest.json").read_text())
        self.assertEqual(manifest["status"],"interrupted_or_failed")
    @unittest.skipUnless(importlib.util.find_spec("optuna") is not None,"Optuna unavailable; real SQLite/TPE test must run locally.")
    def test_real_optuna_sqlite_with_fake_cv_runner(self):
        import optuna
        self.runner.optuna_module=optuna
        self.context["optuna_version"]=optuna.__version__
        self.assertEqual(self.run_fake(),0)
        db=self.config.output_dir/"study.sqlite3"
        self.assertTrue(db.is_file())
        study=optuna.load_study(study_name=self.config.experiment_id,storage=f"sqlite:///{db.as_posix()}")
        self.assertEqual(len(study.trials),3)
        self.assertTrue(all(t.state.name=="COMPLETE" for t in study.trials))


if __name__=="__main__":
    unittest.main(verbosity=2)

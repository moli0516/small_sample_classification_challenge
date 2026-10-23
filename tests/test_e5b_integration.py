"""E5b tests. Fake-model orchestration is distinct from real-XGBoost checks."""
import contextlib
from dataclasses import replace
import importlib.util
import io
import json
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config.xgboost_experiment_config import XGBoostExperimentConfig
from src.data.labelled_data_loader import LabelledDataLoader
from src.modeling.xgboost_pipeline_factory import XGBoostPipelineFactory
from src.validation.saved_split_loader import SavedSplitLoader
from src.validation.paired_comparison import PairedComparison
from src.experiments.xgboost_experiment_runner import XGBoostExperimentRunner


class FakeModel(ClassifierMixin, BaseEstimator):
    """A test double, NOT an XGBoost or scientific model implementation."""
    def __init__(self, **params):
        self.params = params
    def get_params(self, deep=True):
        return dict(self.params, missing=np.nan)
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self
    def predict_proba(self, X):
        score = 1/(1+np.exp(-np.clip(np.asarray(X)[:,0],-20,20)))
        return np.column_stack((1-score,score))
    def get_booster(self):
        return self
    def num_boosted_rounds(self):
        return self.params["n_estimators"]
    def get_score(self, importance_type):
        return {"feature0":1.}
    def save_config(self):
        return json.dumps(dict(test_double=True, parameters=self.params))


class FakeFactory(XGBoostPipelineFactory):
    @staticmethod
    def check_environment():
        return "TEST_DOUBLE_NOT_REAL_XGBOOST"
    def create(self, config, max_depth, n_estimators):
        return Pipeline([("classifier", FakeModel(**self.parameters(config,max_depth,n_estimators)))])


class E5bIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        x,y = make_classification(n_samples=250,n_features=150,n_informative=10,random_state=123)
        self.features = [str(i) for i in range(150)]
        frame = pd.DataFrame(x,columns=self.features)
        frame.insert(0,"id",[f"{i:04d}" for i in range(250)])
        frame["target"] = y
        self.train = self.root/"train.csv"
        frame.to_csv(self.train,index=False)
        self.y = y
        records=[]
        for seed in range(10):
            for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)):
                records.append(dict(repeat=seed,split_seed=seed,fold=fold,
                                    train_indices=tr.tolist(),valid_indices=va.tolist()))
        self.wrapper = dict(index_type="zero_based_row_position",n_rows=250,n_splits=5,
                            split_seeds=list(range(10)),feature_names=self.features,splits=records)
        self.splits,self.sparse = self.root/"leader.json",self.root/"sparse.json"
        self.write(self.splits,self.wrapper)
        self.write(self.sparse,self.wrapper)
        self.references = self.root/"references.csv"
        pd.DataFrame([dict(C=c,repeat=r,split_seed=r,n_folds=5,mean_auc=.8 if c==1 else .79)
                      for c in [.1,1.] for r in range(10)]).to_csv(self.references,index=False)
        self.config = XGBoostExperimentConfig(train_path=self.train,split_indices_path=self.splits,
                    sparse_split_indices_path=self.sparse,reference_scores_path=self.references,
                    output_dir=self.root/"results")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(path,value):
        path.write_text(json.dumps(value),encoding="utf-8")

    def loaded(self):
        return LabelledDataLoader().load(self.config)

    def fold(self, runner=None):
        x,y,ids,_ = self.loaded()
        runner = runner or XGBoostExperimentRunner(factory=FakeFactory())
        return runner._fit_fold(self.config,x,y,ids,self.wrapper["splits"][0],"test",1,50)

    def test_default_grid_and_300_fits(self):
        self.assertEqual(len(list(self.config.candidates())),6)
        self.assertEqual(len(list(self.config.candidates()))*50,300)

    def test_config_unknown_fields_rejected(self):
        path=self.root/"bad.json"
        self.write(path,{"unknown":1})
        with self.assertRaises(ValueError):
            XGBoostExperimentConfig.from_json(path)

    def test_relative_config_paths_resolve_from_project_root(self):
        path=self.root/"config.json"
        self.write(path,{"train_path":"train.csv","output_dir":"new_results"})
        config=XGBoostExperimentConfig.from_json(path,project_root=self.root)
        self.assertEqual(config.train_path,self.train)
        self.assertEqual(config.output_dir,self.root/"new_results")
        self.assertEqual(config.split_indices_path,self.root/"outputs/results/e4_l1_c_1/split_indices.json")

    def test_invalid_grid_rejected(self):
        for grid in ((True,), (1,1), (), (0,)):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                replace(self.config,max_depth_values=grid)

    def test_training_only_schema_and_string_ids(self):
        x,y,ids,features=self.loaded()
        self.assertEqual(x.shape,(250,150))
        self.assertEqual(ids.iloc[0],"0000")
        self.assertNotIn("target",features)
        self.assertNotIn("id",features)
        self.assertEqual(set(y),{0,1})

    def test_duplicate_csv_header_rejected(self):
        text=self.train.read_text()
        header,body=text.split("\n",1)
        self.train.write_text(header.replace(",1,",",0,",1)+"\n"+body)
        with self.assertRaises(ValueError):
            self.loaded()

    def test_nan_predictor_rejected(self):
        df=pd.read_csv(self.train)
        df.loc[0,"0"]=np.nan
        df.to_csv(self.train,index=False)
        with self.assertRaises(ValueError):
            self.loaded()

    def test_native_split_format(self):
        records=SavedSplitLoader().load(self.splits,self.y,self.features)
        self.assertEqual(len(records),50)
        self.assertEqual(len(records[0]["valid_indices"]),50)

    def test_feature_order_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            SavedSplitLoader().load(self.splits,self.y,list(reversed(self.features)))

    def test_overlap_rejected(self):
        record=self.wrapper["splits"][0]
        record["train_indices"].append(record["valid_indices"][0])
        self.write(self.splits,self.wrapper)
        with self.assertRaises(ValueError):
            SavedSplitLoader().load(self.splits,self.y,self.features)

    def test_validation_alias_supported(self):
        for record in self.wrapper["splits"]:
            record["val_indices"]=record.pop("valid_indices")
        self.write(self.splits,self.wrapper)
        records=SavedSplitLoader().load(self.splits,self.y,self.features)
        self.assertEqual(len(records),50)

    def test_duplicate_reference_rejected(self):
        df=pd.read_csv(self.references)
        pd.concat([df,df.iloc[:1]]).to_csv(self.references,index=False)
        with self.assertRaises(ValueError):
            PairedComparison().load_references(self.references)

    def test_different_reference_assignments_rejected(self):
        records=self.wrapper["splits"]
        for fold in range(5):
            for name in ("train_indices","valid_indices"):
                records[fold][name],records[5+fold][name]=records[5+fold][name],records[fold][name]
        self.write(self.sparse,self.wrapper)
        with self.assertRaises(ValueError):
            XGBoostExperimentRunner(factory=FakeFactory()).run(self.config,dry_run=True,confirm_reference_alignment=True)
        self.assertFalse(self.config.output_dir.exists())

    def test_pipeline_factory_no_scaler_and_fixed_parameters(self):
        fake_module=types.SimpleNamespace(XGBClassifier=FakeModel)
        with patch.dict("sys.modules",{"xgboost":fake_module}):
            pipe=XGBoostPipelineFactory().create(self.config,2,100)
        self.assertEqual(list(pipe.named_steps),["classifier"])
        params=pipe.named_steps["classifier"].get_params()
        self.assertEqual(params["max_depth"],2)
        self.assertEqual(params["n_estimators"],100)
        self.assertEqual(params["reg_lambda"],10.)
        self.assertNotIn("early_stopping_rounds",params)

    def test_missing_confirmation_rejected(self):
        with self.assertRaises(ValueError):
            XGBoostExperimentRunner(factory=FakeFactory()).run(self.config,dry_run=True)
        self.assertFalse(self.config.output_dir.exists())

    def test_preflight_and_no_overwrite(self):
        runner=XGBoostExperimentRunner(factory=FakeFactory())
        self.assertEqual(runner.run(self.config,dry_run=True,confirm_reference_alignment=True),0)
        self.assertFalse((self.config.output_dir/"e5b_fold_diagnostics.csv").exists())
        with self.assertRaises(FileExistsError):
            runner.run(self.config,dry_run=True,confirm_reference_alignment=True)

    def test_fold_score_validity(self):
        rec,pred,booster,params=self.fold()
        self.assertTrue(rec["valid"])
        self.assertFalse(rec["review_required"])
        self.assertEqual(rec["rounds_fitted"],50)
        self.assertEqual(len(pred),50)
        self.assertTrue(booster["test_double"])

    def test_failure_not_scored_as_zero(self):
        with patch.object(FakeModel,"fit",side_effect=RuntimeError("injected")):
            rec,_,_,_=self.fold()
        self.assertFalse(rec["valid"])
        self.assertIsNone(rec["auc"])
        self.assertIn("injected",rec["error"])

    def test_invalid_probability_rejected(self):
        with patch.object(FakeModel,"predict_proba",return_value=np.full((50,2),2.)):
            rec,_,_,_=self.fold()
        self.assertFalse(rec["valid"])

    def test_round_count_mismatch_rejected(self):
        with patch.object(FakeModel,"num_boosted_rounds",return_value=10):
            rec,_,_,_=self.fold()
        self.assertFalse(rec["valid"])

    def test_warning_preserved_and_requires_review(self):
        original=FakeModel.fit
        def warned(obj,x,y):
            warnings.warn("review me",UserWarning)
            return original(obj,x,y)
        with patch.object(FakeModel,"fit",warned):
            rec,_,_,_=self.fold()
        self.assertTrue(rec["valid"])
        self.assertTrue(rec["review_required"])
        self.assertEqual(rec["warning_count"],1)

    def test_full_300_fit_orchestration_with_fake_classifier(self):
        runner=XGBoostExperimentRunner(factory=FakeFactory())
        self.assertEqual(runner.run(self.config,confirm_reference_alignment=True),0)
        out=self.config.output_dir
        diag=pd.read_csv(out/"e5b_fold_diagnostics.csv")
        comparison=pd.read_csv(out/"e5b_comparison.csv")
        repeated=pd.read_csv(out/"e5b_paired_repeat_scores.csv")
        self.assertEqual(len(diag),300)
        self.assertEqual(len(comparison),6)
        self.assertEqual(len(repeated),60)
        for _,r in repeated.iterrows():
            f=diag[(diag["experiment_id"]==r["experiment_id"]) & (diag["repeat"]==r["repeat"])]
            self.assertAlmostEqual(r["mean_auc"],f["auc"].mean())
            self.assertAlmostEqual(r["difference_vs_e4_leader"],r["mean_auc"]-.8)
        decision=json.loads((out/"e5b_decision.json").read_text())
        self.assertEqual(decision["e5b_leader"],"e5b_xgboost_depth_1_trees_50")
        for name,_,_ in self.config.candidates():
            self.assertEqual(len(pd.read_csv(out/name/"validation_probabilities.csv")),2500)
            self.assertEqual(len(pd.read_csv(out/name/"fold_scores.csv")),50)
            self.assertEqual(len(pd.read_csv(out/name/"repeat_scores.csv")),10)
            params=json.loads((out/name/"estimator_parameters.json").read_text())
            self.assertEqual(params["missing"],"NaN")
        manifest=json.loads((out/"run_manifest.json").read_text())
        self.assertEqual(manifest["status"],"completed")
        self.assertEqual(manifest["completed_fits"],300)
        self.assertEqual(manifest["versions"]["xgboost"],"TEST_DOUBLE_NOT_REAL_XGBOOST")

    def test_failed_candidate_not_ranked_from_successful_folds(self):
        config=replace(self.config,max_depth_values=(1,),n_estimators_values=(50,))
        with patch.object(FakeModel,"fit",side_effect=RuntimeError("injected")):
            code=XGBoostExperimentRunner(factory=FakeFactory()).run(config,confirm_reference_alignment=True)
        self.assertEqual(code,2)
        summary=pd.read_csv(config.output_dir/"e5b_comparison.csv")
        self.assertFalse(summary.loc[0,"eligible"])
        self.assertNotIn("mean_auc",summary)
        self.assertFalse((config.output_dir/next(config.candidates())[0]/"fold_scores.csv").exists())

    def test_interruption_recorded(self):
        with patch.object(FakeModel,"fit",side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                XGBoostExperimentRunner(factory=FakeFactory()).run(self.config,confirm_reference_alignment=True)
        manifest=json.loads((self.config.output_dir/"run_manifest.json").read_text())
        self.assertEqual(manifest["status"],"interrupted_or_failed")

    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None,"Real XGBoost unavailable; run locally after installation.")
    def test_real_xgboost_one_fold(self):
        rec,pred,booster,params=self.fold(XGBoostExperimentRunner())
        self.assertTrue(rec["valid"],rec["error"])
        self.assertEqual(rec["rounds_fitted"],50)
        self.assertEqual(len(pred),50)
        self.assertIsInstance(booster,dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)

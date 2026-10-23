"""Freeze/export safeguards. Fake models are not real XGBoost training."""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission


class FakeModel:
    test_version="TEST_DOUBLE_NOT_REAL_XGBOOST"
    calls=[]
    def __init__(self,**params):self.params=params
    def fit(self,x,y):
        self.classes_=np.array([0,1]);self.columns=list(x.columns)
        self.gains=dict(zip(x.columns,map(float,abs(np.asarray(x).T@(np.asarray(y)-np.mean(y))))))
        type(self).calls.append(list(x.columns))
        return self
    def get_booster(self):return self
    def num_boosted_rounds(self):return self.params.get("n_estimators",50)
    def get_score(self,importance_type):return self.gains
    def predict_proba(self,x):
        score=1/(1+np.exp(-np.clip(np.asarray(x).mean(axis=1),-20,20)))
        return np.column_stack((1-score,score))
    def save_model(self,path):
        Path(path).write_text(json.dumps(dict(params=self.params,columns=self.columns,gains=self.gains)))
    def load_model(self,path):
        raw=json.loads(Path(path).read_text());self.params=raw["params"];self.columns=raw["columns"];self.gains=raw["gains"]
        self.classes_=np.array([0,1])


class FakeSelector:
    def __init__(self,k):self.k=k
    def fit_from_booster(self,booster,features):
        self.features=list(features);gains=booster.get_score("total_gain")
        self.ranked=sorted(range(len(features)),key=lambda i:(-gains.get(features[i],0),i))[:self.k]
        self.selected=[features[i] for i in sorted(self.ranked)]
        return self
    def transform(self,x):return x[self.selected].copy()
    def get_feature_names_out(self):return np.array(self.selected)
    def ranked_features(self):return [self.features[i] for i in self.ranked]


class FreezeSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.service=FrozenXGBoostSubmission(model_type=FakeModel)
        self.features=["a","b","c","d"]
        rng=np.random.default_rng(42)
        self.x=pd.DataFrame(rng.normal(size=(250,4)),columns=self.features)
        self.y=pd.Series(np.tile([0,1],125));self.train=self.root/"train.csv"
        frame=self.x.copy();frame.insert(0,"id",[str(i) for i in range(250)]);frame["target"]=self.y
        frame.to_csv(self.train,index=False)
        self.test=self.root/"test.csv";self.template=self.root/"template.csv"
        self.test_frame=pd.DataFrame(rng.normal(size=(12,4)),columns=self.features)
        self.test_frame.insert(0,"id",[f"{i:04d}" for i in range(12)])
        self.test_frame.to_csv(self.test,index=False)
        self.template_frame=pd.DataFrame(dict(id=list(reversed(self.test_frame.id.tolist())),target=[0.]*12))
        self.template_frame.to_csv(self.template,index=False)
        self.frozen=self.root/"frozen";self.frozen.mkdir()
        self.write(self.frozen/"base_config.json",{})
        self.write(self.frozen/"frozen_candidate.json",dict(status="frozen_development_procedure",candidate_version="fixture_v1",
            feature_order=self.features,training_sha256=self.service._sha(self.train),estimator_parameters=dict(n_estimators=50,max_depth=4),
            k=2,versions=self.service._versions(),default_test_path=str(self.test),default_template_path=str(self.template),
            id_column="id",target_column="target",expected_test_rows=12))
        self.rehash()
        FakeModel.calls=[]
    def tearDown(self):self.temp.cleanup()
    @staticmethod
    def write(path,obj):path.write_text(json.dumps(obj),encoding="utf-8")
    def rehash(self):
        self.write(self.frozen/"snapshot_sha256.json",{p.name:self.service._sha(p) for p in self.frozen.iterdir()
                                                     if p.is_file() and p.name!="snapshot_sha256.json"})
    def dependency_modules(self):
        config=types.ModuleType("src.config.xgboost_experiment_config")
        config.XGBoostExperimentConfig=types.SimpleNamespace(from_json=lambda path:types.SimpleNamespace(train_path=self.train))
        loader=types.ModuleType("src.data.labelled_data_loader")
        x,y,features=self.x,self.y,self.features
        class Loader:
            def load(self,base):return x,y,pd.Series(range(250)),features
        loader.LabelledDataLoader=Loader
        selector=types.ModuleType("src.features.gain_top_k_selector");selector.GainTopKSelector=FakeSelector
        return {"src.config.xgboost_experiment_config":config,"src.data.labelled_data_loader":loader,
                "src.features.gain_top_k_selector":selector}
    def predict(self,**kwargs):
        with patch.dict("sys.modules",self.dependency_modules()):
            return self.service.refit_predict(self.frozen,self.root/"prediction",**kwargs)
    def test_snapshot_checksums(self):
        root,candidate=self.service._verify_snapshot(self.frozen)
        self.assertEqual(root,self.frozen);self.assertEqual(candidate["k"],2)
    def test_modified_snapshot_rejected(self):
        (self.frozen/"base_config.json").write_text('{"modified":true}')
        with self.assertRaises(ValueError):self.service._verify_snapshot(self.frozen)
    def test_template_id_order_alignment(self):
        result=self.service._align(pd.Series(["a","b"]),np.array([.2,.8]),pd.DataFrame({"id":["b","a"],"target":[0,0]}),"id","target")
        self.assertEqual(result.target.tolist(),[.8,.2])
    def test_unmatched_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.service._align(pd.Series(["a"]),np.array([.2]),pd.DataFrame({"id":["b"]}),"id","target")
    def test_probability_range_rejected(self):
        with self.assertRaises(ValueError):
            self.service._align(pd.Series(["a"]),np.array([2.]),pd.DataFrame({"id":["a"]}),"id","target")
    def test_csv_schema_and_leading_zero_ids(self):
        x,ids,template=self.service._test_inputs(self.test,self.template,self.features,"id","target",12)
        self.assertEqual(ids.iloc[0],"0000");self.assertEqual(template.id.iloc[0],"0011")
    def test_duplicate_ids_rejected(self):
        self.test_frame.loc[1,"id"]=self.test_frame.loc[0,"id"];self.test_frame.to_csv(self.test,index=False)
        with self.assertRaises(ValueError):self.service._test_inputs(self.test,self.template,self.features,"id","target",12)
    def test_test_schema_rejected(self):
        self.test_frame.drop(columns="a").to_csv(self.test,index=False)
        with self.assertRaises(ValueError):self.service._test_inputs(self.test,self.template,self.features,"id","target",12)
    def test_preflight_no_training(self):
        result=self.predict(dry_run=True)
        self.assertEqual(result["fits_performed"],0);self.assertFalse(FakeModel.calls)
        self.assertFalse((self.root/"prediction").exists())
    def test_changed_training_data_rejected(self):
        self.train.write_text("changed")
        with self.assertRaises(ValueError):self.predict(dry_run=True)
    def test_export_and_model_reload_with_test_double(self):
        result=self.predict()
        self.assertEqual(result["status"],"ready_for_manual_submission")
        self.assertEqual(len(FakeModel.calls),2)
        self.assertEqual([len(c) for c in FakeModel.calls],[4,2])
        output=self.root/"prediction"
        frame=pd.read_csv(output/"submission.csv",dtype={"id":"string"})
        self.assertEqual(frame.id.tolist(),self.template_frame.id.tolist())
        self.assertEqual(list(frame),["id","target"])
        manifest=json.loads((output/"prediction_manifest.json").read_text())
        self.assertTrue(manifest["saved_and_reloaded_models_verified"])
        self.assertFalse(manifest["kaggle_upload_performed"])
    def test_existing_prediction_directory_rejected(self):
        (self.root/"prediction").mkdir()
        with self.assertRaises(FileExistsError):self.predict(dry_run=True)
    def test_freeze_requires_confirmation(self):
        with self.assertRaises(ValueError):self.service.freeze(self.root,20,self.root/"new_frozen")
    def test_freeze_requires_completed_extension(self):
        source=self.root/"source";source.mkdir();self.write(source/"run_manifest.json",dict(status="running"))
        with self.assertRaises(ValueError):self.service._verify_completed_extension(source,20)

    def test_full_freeze_snapshot_with_source_adapter_mocked(self):
        source=self.root/"source";source.mkdir()
        candidate=source/"candidate_k2";candidate.mkdir()
        self.write(source/"run_manifest.json",dict(status="completed",planned_fits=200,completed_fits=200,
            control_auc_reproduced=True,input_and_source_sha256={str(self.train):self.service._sha(self.train)}))
        pd.DataFrame([dict(repeat=r,fold=f,matches=True) for r in range(10) for f in range(5)]).to_csv(source/"control_auc_reproduction.csv",index=False)
        pd.DataFrame([dict(experiment_id="candidate_k2",k=2,eligible=True,valid_folds=50,
            mean_auc=.89,std_auc=.01,mean_difference_vs_trial20_control=.008)]).to_csv(source/"feature_selection_comparison.csv",index=False)
        for name in ("config.json","source_trial_parameters.json","verified_split_indices.json","selected_features.json"):
            self.write(source/name,{})
        for name in ("feature_selection_paired_repeat_scores.csv","reference_repeat_scores.csv","selection_frequencies.csv"):
            (source/name).write_text("fixture\n")
        for name in ("config.json","summary.json"):self.write(candidate/name,{})
        for name in ("fold_scores.csv","repeat_scores.csv"):(candidate/name).write_text("fixture\n")
        @dataclass(frozen=True)
        class SourceConfig:
            output_dir:Path=source
            source_trial_number:int=20
            @staticmethod
            def from_json(path):return SourceConfig()
        context=dict(features=self.features,parameters=dict(n_estimators=50,max_depth=4),
            base=types.SimpleNamespace(train_path=self.train,id_column="id",target_column="target",to_dict=lambda:{}))
        class SourceRunner:
            def _prepare(self,config,confirmed):return context
        module_config=types.ModuleType("src.config.xgboost_feature_selection_config")
        module_config.XGBoostFeatureSelectionConfig=SourceConfig
        module_runner=types.ModuleType("src.experiments.xgboost_feature_selection_runner")
        module_runner.XGBoostFeatureSelectionRunner=SourceRunner
        dest=self.root/"new_snapshot"
        with patch.dict("sys.modules",{"src.config.xgboost_feature_selection_config":module_config,
                "src.experiments.xgboost_feature_selection_runner":module_runner}):
            self.assertEqual(self.service.freeze(source,2,dest,True),dest)
        _,frozen=self.service._verify_snapshot(dest)
        self.assertEqual(frozen["k"],2)
        self.assertEqual(frozen["expected_test_rows"],19750)
        self.assertTrue((dest/"evidence/selected_candidate/fold_scores.csv").exists())
        self.assertFalse(FakeModel.calls)

    def test_float32_predictions_preserve_csv_precision(self):
        class Float32Model:
            classes_=np.array([0,1])
            def predict_proba(self,x):
                score=np.full(len(x),np.float32(.12345679),dtype=np.float32)
                return np.column_stack((1-score,score))
        scores=self.service._probabilities(Float32Model(),self.test_frame[self.features])
        self.assertEqual(scores.dtype,np.float64)
        result=self.service._align(self.test_frame.id,scores,self.template_frame,"id","target")
        path=self.root/"precision.csv";result.to_csv(path,index=False)
        loaded=pd.read_csv(path)
        np.testing.assert_allclose(loaded.target,result.target,rtol=0,atol=1e-12)
    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None,"Real XGBoost unavailable; run UBJSON round-trip locally.")
    def test_real_xgboost_ubjson_roundtrip(self):
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=50,max_depth=4,n_jobs=1,random_state=42).fit(self.x,self.y)
        p=self.root/"real.ubj";model.save_model(p)
        loaded=XGBClassifier();loaded.load_model(p)
        np.testing.assert_allclose(self.service._probabilities(model,self.x),self.service._probabilities(loaded,self.x),rtol=0,atol=1e-12)


if __name__=="__main__":unittest.main(verbosity=2)

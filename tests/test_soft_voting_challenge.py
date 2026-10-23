"""Synthetic core-flow tests; repository adapters/native XGBoost need local validation."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.soft_voting_challenge import SoftVotingChallenge


class Helper:
    @staticmethod
    def _path(path):return Path(path).resolve()
    @staticmethod
    def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @staticmethod
    def _write(path,obj):Path(path).write_text(json.dumps(obj,allow_nan=False),encoding="utf-8")
    @staticmethod
    def _probabilities(model,x):
        if not np.array_equal(model.classes_,[0,1]):raise ValueError("Class order")
        return np.asarray(model.predict_proba(x),dtype=float)[:,1]
    @staticmethod
    def _read_csv(path,id_column):return pd.read_csv(path,dtype={id_column:"string"})
    @staticmethod
    def _align(ids,scores,template,id_column,target_column):
        result=template[[id_column]].copy()
        result[target_column]=result[id_column].map(pd.Series(scores,index=ids.to_numpy()))
        if result[target_column].isna().any():raise ValueError("ID mismatch")
        return result
    def _test_inputs(self,test,template,features,id_column,target_column,expected_rows):
        frame=self._read_csv(test,id_column);target=self._read_csv(template,id_column)
        return frame[features],frame[id_column],target


class FakeTree:
    """JSON test double; its .ubj output is NOT an actual XGBoost model."""
    def __init__(self,**params):self.params=params
    def fit(self,x,y):
        self.classes_=np.array([0,1]);self.columns=list(x.columns)
        self.gains=dict(zip(self.columns,map(float,abs(np.asarray(x).T@(np.asarray(y)-np.mean(y))))))
        return self
    def get_booster(self):return self
    def num_boosted_rounds(self):return self.params["n_estimators"]
    def predict_proba(self,x):
        p=1/(1+np.exp(-np.clip(np.asarray(x).mean(axis=1),-20,20)))
        return np.column_stack((1-p,p))
    def save_model(self,path):Helper._write(path,dict(params=self.params,columns=self.columns,gains=self.gains))
    def load_model(self,path):
        data=json.loads(Path(path).read_text());self.params=data["params"];self.columns=data["columns"];self.gains=data["gains"]
        self.classes_=np.array([0,1])


class Selector:
    def __init__(self,k):self.k=k
    def fit_from_booster(self,booster,features):
        ranked=sorted(range(len(features)),key=lambda i:(-booster.gains[features[i]],i))[:self.k]
        self.selected=[features[i] for i in sorted(ranked)];return self
    def transform(self,x):return x[self.selected].copy()
    def get_feature_names_out(self):return np.array(self.selected)


class SoftVotingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.runner=SoftVotingChallenge();self.runner.helper=Helper();self.runner.model_type=FakeTree
        self.rng=np.random.default_rng(42)
        x=pd.DataFrame(self.rng.normal(size=(50,4)),columns=list("abcd"))
        y=pd.Series(np.tile([0,1],25));ids=pd.Series([str(i) for i in range(50)],dtype="string")
        train=self.root/"train.csv";x.assign(target=y).to_csv(train,index=False)
        test=self.root/"test.csv";template=self.root/"template.csv"
        x.iloc[:8].assign(id=[f"{i:04d}" for i in range(8)]).to_csv(test,index=False)
        pd.DataFrame(dict(id=[f"{i:04d}" for i in reversed(range(8))],target=0.)).to_csv(template,index=False)
        splits=[];expected={}
        for r in range(10):
            for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
                splits.append(dict(repeat=r,split_seed=r,fold=f,train_indices=tr.tolist(),valid_indices=va.tolist()))
                rank=FakeTree(n_estimators=50).fit(x.iloc[tr],y.iloc[tr])
                selector=Selector(2).fit_from_booster(rank,list(x))
                tree=FakeTree(n_estimators=50).fit(selector.transform(x.iloc[tr]),y.iloc[tr])
                expected[(r,f)]=roc_auc_score(y.iloc[va],tree.predict_proba(selector.transform(x.iloc[va]))[:,1])
        candidate=dict(k=2,estimator_parameters=dict(n_estimators=50),default_test_path=str(test),
                       default_template_path=str(template),expected_test_rows=8)
        self.c=dict(output=self.root/"cv",frozen=self.root/"frozen",candidate=candidate,versions=dict(test_double=True),
            base=types.SimpleNamespace(id_column="id",target_column="target"),x=x,y=y,ids=ids,features=list(x),
            linear=types.SimpleNamespace(to_dict=lambda:dict(fixture=True)),l1_path=self.root/"l1.json",splits=splits,
            expected=pd.Series(expected),hashes={str(train):Helper._sha(train)})
        def prepare(frozen,l1,output):
            if Path(output).exists():raise FileExistsError("Output exists")
            return dict(self.c,output=Path(output))
        self.runner._prepare=prepare
        self.runner._linear_model=lambda config:make_pipeline(StandardScaler(),LogisticRegression(solver="liblinear",random_state=42))
        selector_module=types.ModuleType("src.features.gain_top_k_selector");selector_module.GainTopKSelector=Selector
        helper_module=types.ModuleType("src.submission.frozen_xgboost_submission");helper_module.FrozenXGBoostSubmission=Helper
        self.modules={"src.features.gain_top_k_selector":selector_module,"src.submission.frozen_xgboost_submission":helper_module}
    def tearDown(self):self.temp.cleanup()
    def run_cv(self,dry=False):
        with patch.dict("sys.modules",self.modules),contextlib.redirect_stdout(io.StringIO()):
            return self.runner.cv("unused","unused",self.c["output"],dry)
    def test_blend_equal_average(self):np.testing.assert_allclose(self.runner.blend([.2,.8],[.4,.6]),[.3,.7])
    def test_float32_promoted(self):self.assertEqual(self.runner.blend(np.array([.2],dtype=np.float32),[.4]).dtype,np.float64)
    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):self.runner.blend([.2],[.2,.3])
    def test_empty_predictions(self):
        with self.assertRaises(ValueError):self.runner.blend([],[])
    def test_matrix_rejected(self):
        with self.assertRaises(ValueError):self.runner.blend([[.2]],[[.3]])
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):self.runner.blend([float("nan")],[.2])
    def test_range_rejected(self):
        with self.assertRaises(ValueError):self.runner.blend([1.2],[.2])
    def summary_frame(self,values):
        return pd.DataFrame([dict(repeat=i,split_seed=i,l1=.7,xgboost=.8,ensemble=v) for i,v in enumerate(values)])
    def test_positive_consistent_promotion(self):
        repeats,summary=self.runner._summary(self.summary_frame([.81]*10))
        self.assertTrue(summary["promote_ensemble"]);self.assertEqual(summary["wins_vs_xgboost"],10)
    def test_positive_mean_but_few_wins_not_promoted(self):
        _,summary=self.runner._summary(self.summary_frame([.99]*3+[.79]*7))
        self.assertGreater(summary["mean_difference_vs_xgboost"],0);self.assertFalse(summary["promote_ensemble"])
    def test_weaker_blend_not_promoted(self):
        _,summary=self.runner._summary(self.summary_frame([.79]*10));self.assertFalse(summary["promote_ensemble"])
    def test_cv_preflight_no_output(self):
        result=self.run_cv(True);self.assertEqual(result["planned_fits"],150);self.assertFalse(self.c["output"].exists())
    def test_synthetic_cv_and_control_reproduction(self):
        self.run_cv()
        manifest=json.loads((self.c["output"]/"run_manifest.json").read_text())
        self.assertEqual(manifest["status"],"completed");self.assertEqual(manifest["completed_folds"],50)
        self.assertTrue(manifest["control_auc_reproduced"]);self.assertFalse(manifest["test_opened"])
        predictions=pd.read_csv(self.c["output"]/"validation_probabilities.csv")
        self.assertEqual(len(predictions),500)
        self.assertEqual(predictions.groupby(["repeat","row_index"]).size().max(),1)
    def test_control_mismatch_stops(self):
        self.c["expected"].iloc[0]=-1
        with self.assertRaises(RuntimeError):self.run_cv()
        self.assertEqual(json.loads((self.c["output"]/"run_manifest.json").read_text())["status"],"failed_or_interrupted")
    def test_warning_stops(self):
        original=self.runner._fit
        def fit(*args):warnings.warn("review required",UserWarning);return original(*args)
        self.runner._fit=fit
        with self.assertRaises(RuntimeError):self.run_cv()
        self.assertTrue(json.loads((self.c["output"]/"warnings.json").read_text()))
    def make_completed_cv(self,promote):
        run=self.root/"evaluated";run.mkdir()
        settings=dict(frozen_dir=str(self.c["frozen"]),l1_config_path=str(self.c["l1_path"]),
                      versions=self.c["versions"],input_and_code_sha256=self.c["hashes"])
        Helper._write(run/"config.json",settings);Helper._write(run/"summary.json",dict(promote_ensemble=promote))
        Helper._write(run/"run_manifest.json",dict(status="completed",completed_folds=50,control_auc_reproduced=True,
            artifact_sha256={p.name:Helper._sha(p) for p in run.iterdir()}))
        return run
    def test_nonpromoted_prediction_rejected(self):
        run=self.make_completed_cv(False)
        with patch.dict("sys.modules",self.modules),self.assertRaises(ValueError):
            self.runner.predict(run,self.root/"prediction")
    def test_changed_cv_artifact_rejected(self):
        run=self.make_completed_cv(True);Helper._write(run/"summary.json",dict(changed=True))
        with patch.dict("sys.modules",self.modules),self.assertRaises(ValueError):self.runner.predict(run,self.root/"prediction")
    def test_synthetic_refit_export_and_reload(self):
        run=self.make_completed_cv(True);out=self.root/"prediction"
        with patch.dict("sys.modules",self.modules):result=self.runner.predict(run,out)
        self.assertEqual(result["status"],"ready_for_manual_submission")
        frame=pd.read_csv(out/"submission.csv",dtype={"id":"string"})
        self.assertEqual(frame.id.tolist(),[f"{i:04d}" for i in reversed(range(8))])
        manifest=json.loads((out/"prediction_manifest.json").read_text())
        self.assertTrue(manifest["saved_and_reloaded_models_verified"]);self.assertEqual(manifest["fits_performed"],3)
    def test_prediction_preflight_no_output(self):
        run=self.make_completed_cv(True);out=self.root/"prediction"
        with patch.dict("sys.modules",self.modules):result=self.runner.predict(run,out,dry_run=True)
        self.assertEqual(result["fits_performed"],0);self.assertFalse(out.exists())
    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None,"Native XGBoost unavailable here; run locally.")
    def test_real_xgboost_serialization(self):
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=5,n_jobs=1,random_state=42).fit(self.c["x"],self.c["y"])
        path=self.root/"native.ubj";model.save_model(path);loaded=XGBClassifier();loaded.load_model(path)
        np.testing.assert_allclose(model.predict_proba(self.c["x"]),loaded.predict_proba(self.c["x"]),rtol=0,atol=1e-12)


if __name__=="__main__":unittest.main()

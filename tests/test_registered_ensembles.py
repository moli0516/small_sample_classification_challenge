"""Registry, cache guards and synthetic CV/export. Native integrations run locally."""
import contextlib
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.modeling.model_registry import ModelRegistry, RegisteredModel, soft_vote
from src.experiments.registered_ensembles import RegisteredEnsembles


class FakeModel(RegisteredModel):
    axis = 0
    def __init__(self, *, context=None, parameters=None):pass
    def fit(self, X, y):
        self._remember(X); self.training_rows_ = len(X)
        return self
    @property
    def classes_(self):return np.array([0,1])
    def predict_proba(self, X):
        self._check(X)
        p=1/(1+np.exp(-np.clip(np.asarray(X)[:,self.axis]*3,-20,20)))
        return np.column_stack((1-p,p))


class FakeL1(FakeModel):axis=0
class FakeXGB(FakeModel):axis=1
class FakeLGBM(FakeModel):axis=2


class LinearConfig:
    def to_dict(self):return dict(fixture=True)


class Helper:
    @staticmethod
    def _path(path):return Path(path).resolve()
    @staticmethod
    def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @staticmethod
    def _write(path,obj):Path(path).write_text(json.dumps(obj,allow_nan=False),encoding="utf-8")
    @staticmethod
    def _read_csv(path,id_column):return pd.read_csv(path,dtype={id_column:"string"})
    @staticmethod
    def _align(ids,scores,target,id_column,target_column):
        f=target[[id_column]].copy();f[target_column]=f[id_column].map(pd.Series(scores,index=ids.to_numpy()))
        if f[target_column].isna().any():raise ValueError("Unmatched IDs")
        return f
    def _test_inputs(self,test,template,features,id_column,target_column,expected_rows):
        x=self._read_csv(test,id_column);t=self._read_csv(template,id_column)
        return x[features],x[id_column],t


class RegistryTests(unittest.TestCase):
    def test_builtin_names(self):self.assertTrue({"l1_pipeline","xgb_topk","lgbm_classifier"}<=set(ModelRegistry.names()))
    def test_unknown_model(self):
        with self.assertRaises(ValueError):ModelRegistry.create("does_not_exist")
    def test_duplicate_registration(self):
        ModelRegistry.names()
        with self.assertRaises(ValueError):ModelRegistry.register("l1_pipeline")(FakeL1)
    def test_invalid_base_class(self):
        with self.assertRaises(TypeError):ModelRegistry.register("invalid")(object)
    def test_equal_pair(self):np.testing.assert_allclose(soft_vote({"a":[.2],"b":[.8]},["a","b"],[1,1]),[.5])
    def test_equal_triple(self):np.testing.assert_allclose(soft_vote({"a":[.2],"b":[.5],"c":[.8]},["a","b","c"],[1,1,1]),[.5])
    def test_huge_weights(self):np.testing.assert_allclose(soft_vote({"a":[.2],"b":[.8]},["a","b"],[1e308,1e308]),[.5])
    def test_invalid_weight_sets(self):
        for weights in ([0,0],[-1,2],[float("nan"),1],[float("inf"),1]):
            with self.subTest(weights=weights),self.assertRaises(ValueError):soft_vote({"a":[.2],"b":[.8]},["a","b"],weights)
    def test_bad_probability(self):
        for p in ([],[1.2],[float("nan")]):
            with self.subTest(p=p),self.assertRaises(ValueError):soft_vote({"a":p},["a"],[1])
    def test_missing_member(self):
        with self.assertRaises(ValueError):soft_vote({"a":[.2]},["missing"],[1])
    def test_duplicate_member(self):
        with self.assertRaises(ValueError):soft_vote({"a":[.2]},["a","a"],[1,1])
    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):soft_vote({"a":[.2],"b":[.2,.3]},["a","b"],[1,1])
    def test_feature_order_guard(self):
        x=pd.DataFrame({"a":[1,2],"b":[2,1]});model=FakeL1().fit(x,[0,1])
        with self.assertRaises(ValueError):model.positive_probability(x[["b","a"]])
    def test_class_orientation_guard(self):
        class Reverse(FakeL1):
            @property
            def classes_(self):return np.array([1,0])
        x=pd.DataFrame({"a":[1,2]});model=Reverse().fit(x,[0,1])
        with self.assertRaises(ValueError):model.positive_probability(x)
    def test_lazy_optional_imports_in_subprocess(self):
        code='from src.modeling.model_registry import ModelRegistry; import sys; ModelRegistry.names(); assert "lightgbm" not in sys.modules and "xgboost" not in sys.modules'
        result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        root=Path(__file__).resolve().parents[1]
        self.cfg=json.loads((root/"configs/registry_ensembles.json").read_text())
        self.cfg["output_dir"]=str(self.root/"cv");self.cfg["reuse_cv_run"]=str(self.root/"cache")
        self.config=self.root/"config.json";Helper._write(self.config,self.cfg)
        rng=np.random.default_rng(42);x=pd.DataFrame(rng.normal(size=(50,4)),columns=list("abcd"))
        y=pd.Series((x.c>x.c.median()).astype(int));ids=pd.Series([f"{i:04d}" for i in range(50)],dtype="string")
        train=self.root/"train.csv";x.assign(id=ids,target=y).to_csv(train,index=False)
        test=self.root/"test.csv";template=self.root/"template.csv"
        x.iloc[:8].assign(id=[f"{i:04d}" for i in range(8)]).to_csv(test,index=False)
        pd.DataFrame(dict(id=[f"{i:04d}" for i in reversed(range(8))],target=0.)).to_csv(template,index=False)
        splits,records,folds,expected=[],[],[],{}
        for r in range(10):
            for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
                key=dict(repeat=r,split_seed=r,fold=f)
                splits.append(dict(key,train_indices=tr.tolist(),valid_indices=va.tolist()))
                a=FakeL1().fit(x.iloc[tr],y.iloc[tr]).positive_probability(x.iloc[va])
                b=FakeXGB().fit(x.iloc[tr],y.iloc[tr]).positive_probability(x.iloc[va])
                mix=(a+b)/2
                auc={n:roc_auc_score(y.iloc[va],p) for n,p in (("l1",a),("xgboost",b),("ensemble",mix))}
                folds.append(dict(key,**auc));expected[(r,f)]=auc["xgboost"]
                for j,i in enumerate(va):records.append(dict(key,row_index=int(i),id=ids.iloc[i],target=int(y.iloc[i]),l1=a[j],xgboost=b[j],ensemble=mix[j]))
        self.c=dict(x=x,y=y,ids=ids,features=list(x),splits=splits,expected=pd.Series(expected),
            hashes={str(train):Helper._sha(train)},versions=dict(fixture=True),linear=LinearConfig(),
            base=types.SimpleNamespace(id_column="id",target_column="target"),
            candidate=dict(default_test_path=str(test),default_template_path=str(template),expected_test_rows=8,
                           estimator_parameters=dict(n_estimators=50),k=2),cfg=self.cfg,config_path=self.config,no_cache=False)
        cache=self.root/"cache";cache.mkdir()
        pd.DataFrame(records).to_csv(cache/"validation_probabilities.csv",index=False)
        pd.DataFrame(folds).to_csv(cache/"fold_scores.csv",index=False)
        Helper._write(cache/"summary.json",dict(fixture=True))
        Helper._write(cache/"config.json",dict(input_and_code_sha256=self.c["hashes"],versions=self.c["versions"]))
        self.rehash_cache()
        self.runner=RegisteredEnsembles();self.runner.helper=Helper()
        self.c["cache"]=pd.DataFrame(records).set_index(["repeat","split_seed","fold","row_index"])
        self.patch=patch.dict(ModelRegistry._models,{"l1_pipeline":FakeL1,"xgb_topk":FakeXGB,"lgbm_classifier":FakeLGBM})
        ModelRegistry.names();self.patch.start()
        def prepare(path,output=None,no_cache=False):
            destination=Path(output or self.cfg["output_dir"])
            if destination.exists():raise FileExistsError("Output exists")
            return dict(self.c,output=destination,no_cache=no_cache,cache=None if no_cache else self.c["cache"])
        self.runner._prepare=prepare
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def rehash_cache(self):
        cache=self.root/"cache"
        Helper._write(cache/"run_manifest.json",dict(status="completed",completed_folds=50,control_auc_reproduced=True,
            artifact_sha256={p.name:Helper._sha(p) for p in cache.iterdir() if p.name!="run_manifest.json"}))
    def cv(self,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return self.runner.cv(self.config,**kwargs)
    def test_config_validation(self):self.assertEqual(len(self.runner.validate(deepcopy(self.cfg))["ensembles"]),4)
    def test_invalid_config_weights(self):
        cfg=deepcopy(self.cfg);cfg["ensembles"]["lgbm_l1"]["weights"]=[0,0]
        with self.assertRaises(ValueError):self.runner.validate(cfg)
    def test_cache_verified(self):
        indexed,paths=self.runner._cache(self.c,self.root/"cache")
        self.assertEqual(len(indexed),500);self.assertGreater(len(paths),3)
    def test_cache_hash_change_rejected(self):
        (self.root/"cache/summary.json").write_text("changed")
        with self.assertRaises(ValueError):self.runner._cache(self.c,self.root/"cache")
    def test_cache_wrong_ids_rejected(self):
        p=self.root/"cache/validation_probabilities.csv";df=pd.read_csv(p,dtype={"id":"string"});df.loc[0,"id"]="WRONG";df.to_csv(p,index=False);self.rehash_cache()
        with self.assertRaises(ValueError):self.runner._cache(self.c,self.root/"cache")
    def test_cache_wrong_training_hash_rejected(self):
        self.c["hashes"]={"changed":"bad"}
        with self.assertRaises(ValueError):self.runner._cache(self.c,self.root/"cache")
    def test_cache_duplicate_row_rejected(self):
        p=self.root/"cache/validation_probabilities.csv";df=pd.read_csv(p,dtype={"id":"string"});df.iloc[1]=df.iloc[0];df.to_csv(p,index=False);self.rehash_cache()
        with self.assertRaises(ValueError):self.runner._cache(self.c,self.root/"cache")
    def test_preflight_50_new_fits(self):
        result=self.cv(dry_run=True);self.assertEqual(result["new_fits"],50);self.assertFalse((self.root/"cv").exists())
    def test_no_cache_200_fits(self):self.assertEqual(self.cv(dry_run=True,no_cache=True)["new_fits"],200)
    def test_synthetic_cv_seven_candidates(self):
        result=self.cv();self.assertEqual(len(result["comparison"]),7)
        table=pd.read_csv(self.root/"cv/comparison.csv");self.assertEqual(len(table),7)
        manifest=json.loads((self.root/"cv/run_manifest.json").read_text())
        self.assertEqual(manifest["fits_performed"],50);self.assertFalse(manifest["test_opened"])
        self.assertIsNotNone(result["decision"]["selected"])
    def test_changed_input_stops(self):
        path=next(iter(self.c["hashes"]));Path(path).write_text("changed")
        with self.assertRaises(RuntimeError):self.cv()
    def test_prediction_reload_and_template_order(self):
        result=self.cv();self.assertIsNotNone(result["decision"]["selected"])
        helper_module=types.ModuleType("src.submission.frozen_xgboost_submission");helper_module.FrozenXGBoostSubmission=Helper
        with patch.dict("sys.modules",{"src.submission.frozen_xgboost_submission":helper_module}):
            result=self.runner.predict(self.root/"cv",self.root/"prediction")
        self.assertEqual(result["status"],"ready_for_manual_submission")
        data=pd.read_csv(self.root/"prediction/submission.csv",dtype={"id":"string"})
        self.assertEqual(data.id.tolist(),[f"{i:04d}" for i in reversed(range(8))])
    def test_nonpromoted_candidate_rejected(self):
        self.cv()
        helper_module=types.ModuleType("src.submission.frozen_xgboost_submission");helper_module.FrozenXGBoostSubmission=Helper
        with patch.dict("sys.modules",{"src.submission.frozen_xgboost_submission":helper_module}),self.assertRaises(ValueError):
            self.runner.predict(self.root/"cv",self.root/"prediction",candidate="l1_xgb")
    @unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None,"Native LightGBM unavailable; run locally.")
    def test_native_lightgbm_wrapper(self):
        from src.modeling.registered_models import LightGBMModel
        model=LightGBMModel(parameters=dict(n_estimators=5,num_leaves=3,min_child_samples=5,n_jobs=1,verbosity=-1))
        model.fit(self.c["x"],self.c["y"]);p=model.positive_probability(self.c["x"])
        self.assertEqual(len(p),50)
    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None,"Native XGBoost unavailable; run locally.")
    def test_native_xgboost_roundtrip(self):
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=5,n_jobs=1).fit(self.c["x"],self.c["y"])
        path=self.root/"native.ubj";model.save_model(path);loaded=XGBClassifier();loaded.load_model(path)
        np.testing.assert_allclose(model.predict_proba(self.c["x"]),loaded.predict_proba(self.c["x"]),rtol=0,atol=1e-12)


if __name__=="__main__":unittest.main()

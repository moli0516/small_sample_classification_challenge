"""Feature-selection tests; orchestration uses explicit test doubles."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin,BaseEstimator
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config.xgboost_feature_selection_config import XGBoostFeatureSelectionConfig
from src.features.gain_top_k_selector import GainTopKSelector
from src.experiments.xgboost_feature_selection_runner import XGBoostFeatureSelectionRunner


class FakeGainBooster:
    def __init__(self,scores):self.scores=scores
    def get_score(self,importance_type):return self.scores


class FakeModel(ClassifierMixin,BaseEstimator):
    calls=[]
    def __init__(self,n_estimators=50,max_depth=4,random_state=42):
        self.n_estimators=n_estimators;self.max_depth=max_depth;self.random_state=random_state
    def fit(self,x,y):
        self.classes_=np.array([0,1]);self.columns_=list(x.columns)
        self.gains_=dict(zip(x.columns,abs(np.asarray(x).T@(np.asarray(y)-np.mean(y)))))
        type(self).calls.append(dict(indices=list(x.index),features=list(x.columns)))
        return self
    def predict_proba(self,x):
        z=np.asarray(x).mean(axis=1)
        score=1/(1+np.exp(-np.clip(z,-20,20)))
        return np.column_stack((1-score,score))
    def get_booster(self):return self
    def num_boosted_rounds(self):return self.n_estimators
    def get_score(self,importance_type):return self.gains_
    def save_config(self):return json.dumps(dict(test_double=True,n_estimators=self.n_estimators))


class FakeFactory:
    def create(self,config,depth,trees):
        return Pipeline([("classifier",FakeModel(n_estimators=trees,max_depth=depth,random_state=config.model_seed))])


class FakeResult:
    """Minimal result contract for adapter tests, not replacement production code."""
    def __init__(self,frame):
        self.fold_scores=frame[["repeat","split_seed","fold","train_rows","valid_rows","valid_auc"]].copy()
    def repeat_scores(self):
        return self.fold_scores.groupby(["repeat","split_seed"],as_index=False).agg(mean_auc=("valid_auc","mean"),n_folds=("fold","count"))
    def summary(self):
        s=self.repeat_scores()["mean_auc"]
        return dict(mean_auc=float(s.mean()),std_auc=float(s.std(ddof=1)),min_auc=float(s.min()),max_auc=float(s.max()),n_model_fits=len(self.fold_scores))
    def save(self,path):
        path.mkdir(exist_ok=True,parents=True)
        self.fold_scores.to_csv(path/"fold_scores.csv",index=False)
        self.repeat_scores().to_csv(path/"repeat_scores.csv",index=False)
        (path/"summary.json").write_text(json.dumps(self.summary()))


class FeatureSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.config=XGBoostFeatureSelectionConfig(output_dir=self.root/"results",source_study_dir=self.root/"study")
        x,y=make_classification(n_samples=250,n_features=150,n_informative=10,random_state=123)
        features=[str(i) for i in range(150)]
        x=pd.DataFrame(x,columns=features);y=pd.Series(y)
        base=types.SimpleNamespace(model_seed=42)
        splits=[];expected=[]
        factory=FakeFactory()
        for repeat in range(10):
            for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=repeat).split(x,y)):
                splits.append(dict(repeat=repeat,split_seed=repeat,fold=fold,train_indices=tr.tolist(),valid_indices=va.tolist()))
                model=factory.create(base,4,50).fit(x.iloc[tr],y.iloc[tr])
                expected.append(dict(repeat=repeat,fold=fold,valid_auc=roc_auc_score(y.iloc[va],model.predict_proba(x.iloc[va])[:,1])))
        self.sentinel=self.root/"input.txt";self.sentinel.write_text("unchanged")
        self.runner=XGBoostFeatureSelectionRunner(result_type=FakeResult)
        self.context=dict(x=x,y=y,ids=pd.Series([f"{i:04d}" for i in range(250)]),features=features,splits=splits,
            expected=pd.DataFrame(expected).set_index(["repeat","fold"])["valid_auc"],
            references=pd.DataFrame(dict(repeat=range(10),split_seed=range(10),e4_leader_auc=[.8]*10,e4_sparse_auc=[.79]*10)),
            base=base,factory=factory,parameters=dict(max_depth=4,n_estimators=50,random_state=42),
            hashes={str(self.sentinel):self.runner._sha(self.sentinel)},source_mean=float(np.mean([r["valid_auc"] for r in expected])),
            source_candidate="fixture",split_signature="fixture",xgboost_version="TEST_DOUBLE_NOT_REAL_XGBOOST")
        FakeModel.calls=[]
    def tearDown(self):self.temp.cleanup()
    def run_fake(self):
        with patch.object(self.runner,"_prepare",return_value=self.context):
            return self.runner.run(self.config,confirm_reference_alignment=True)
    def test_config_defaults(self):
        self.assertEqual(self.config.k_values,(5,10,20))
        self.assertEqual(self.config.source_trial_number,20)
    def test_paths_resolved(self):
        p=self.root/"config.json";p.write_text('{"source_study_dir":"study","output_dir":"results"}')
        config=XGBoostFeatureSelectionConfig.from_json(p,project_root=self.root)
        self.assertEqual(config.source_study_dir,self.root/"study")
        self.assertEqual(config.output_dir,self.root/"results")
    def test_unknown_config_rejected(self):
        p=self.root/"bad.json";p.write_text('{"global_top_features":[98,112]}')
        with self.assertRaises(ValueError):XGBoostFeatureSelectionConfig.from_json(p)
    def test_invalid_k_rejected(self):
        for values in ((),(0,),(True,),(5,5)):
            with self.subTest(values=values),self.assertRaises(ValueError):replace(self.config,k_values=values)
    def test_selector_preserves_original_order(self):
        selector=GainTopKSelector(2).fit_from_booster(FakeGainBooster({"c":10,"b":8,"a":1}),["b","a","c"])
        self.assertEqual(selector.ranked_features(),["c","b"])
        self.assertEqual(selector.get_feature_names_out().tolist(),["b","c"])
        x=pd.DataFrame([[1,2,3]],columns=["b","a","c"])
        self.assertEqual(selector.transform(x).columns.tolist(),["b","c"])
    def test_selector_zero_fill_and_deterministic_ties(self):
        selector=GainTopKSelector(3).fit_from_booster(FakeGainBooster({"b":4,"c":4}),["a","b","c","d"])
        self.assertEqual(selector.ranked_features(),["b","c","a"])
        np.testing.assert_array_equal(selector.gains_,[0,4,4,0])
    def test_unknown_feature_rejected(self):
        with self.assertRaises(ValueError):GainTopKSelector(1).fit_from_booster(FakeGainBooster({"unknown":2}),["a"])
    def test_negative_gain_rejected(self):
        with self.assertRaises(ValueError):GainTopKSelector(1).fit_from_booster(FakeGainBooster({"a":-1}),["a"])
    def test_transform_schema_mismatch_rejected(self):
        selector=GainTopKSelector(1).fit_from_booster(FakeGainBooster({"a":1}),["a","b"])
        with self.assertRaises(ValueError):selector.transform(pd.DataFrame([[1,2]],columns=["b","a"]))
    def test_selector_fit_interface(self):
        selector=GainTopKSelector(5,estimator=FakeModel()).fit(self.context["x"].iloc[:200],self.context["y"].iloc[:200])
        self.assertEqual(selector.transform(self.context["x"].iloc[200:]).shape,(50,5))
    def test_confirmation_required(self):
        with self.assertRaises(ValueError):self.runner._prepare(self.config,False)
    def test_output_not_overwritten(self):
        self.config.output_dir.mkdir()
        with self.assertRaises(FileExistsError):self.runner._prepare(self.config,True)
    def test_preflight_no_fits(self):
        with patch.object(self.runner,"_prepare",return_value=self.context):
            self.assertEqual(self.runner.run(self.config,dry_run=True,confirm_reference_alignment=True),0)
        self.assertFalse(FakeModel.calls)
    def test_validation_labels_do_not_affect_ranking(self):
        split=self.context["splits"][0]
        tr,va=split["train_indices"],split["valid_indices"]
        x=self.context["x"]
        rec,pred,model,_=self.runner._fit(self.context,split,150,"control",x.iloc[tr],x.iloc[va])
        first=model.named_steps["classifier"].get_score("total_gain")
        self.context["y"].iloc[va]=1-self.context["y"].iloc[va]
        rec,pred,model,_=self.runner._fit(self.context,split,150,"control",x.iloc[tr],x.iloc[va])
        self.assertEqual(first,model.named_steps["classifier"].get_score("total_gain"))
    def test_full_200_fit_orchestration_and_pairing(self):
        self.assertEqual(self.run_fake(),0)
        self.assertEqual(len(FakeModel.calls),200)
        for index,split in enumerate(self.context["splits"]):
            calls=FakeModel.calls[4*index:4*index+4]
            self.assertEqual([len(r["features"]) for r in calls],[150,5,10,20])
            for call in calls:self.assertEqual(call["indices"],split["train_indices"])
        out=self.config.output_dir
        comparison=pd.read_csv(out/"feature_selection_comparison.csv")
        repeats=pd.read_csv(out/"feature_selection_paired_repeat_scores.csv")
        self.assertEqual(len(comparison),4)
        self.assertEqual(len(repeats),40)
        self.assertTrue(comparison["eligible"].all())
        controls=repeats.loc[repeats.k==150].set_index("repeat")["mean_auc"]
        for _,r in repeats.iterrows():
            self.assertAlmostEqual(r["difference_vs_trial20_control"],r["mean_auc"]-controls.loc[r["repeat"]])
        selections=json.loads((out/"selected_features.json").read_text())
        self.assertEqual(len(selections),150)
        for record in selections:self.assertEqual(len(record["selected_features"]),record["k"])
        self.assertEqual(len(pd.read_csv(out/"training_feature_rankings.csv")),7500)
        self.assertEqual(len(pd.read_csv(out/"selection_frequencies.csv")),450)
        manifest=json.loads((out/"run_manifest.json").read_text())
        self.assertEqual(manifest["status"],"completed")
        self.assertEqual(manifest["completed_fits"],200)
        self.assertTrue(manifest["control_auc_reproduced"])
    def test_control_mismatch_stops_before_subset_fits(self):
        self.context["expected"].iloc[0]=0.01
        with self.assertRaises(RuntimeError):self.run_fake()
        self.assertEqual(len(FakeModel.calls),1)
        manifest=json.loads((self.config.output_dir/"run_manifest.json").read_text())
        self.assertEqual(manifest["status"],"interrupted_or_failed")
    def test_input_mutation_stops_before_any_fit(self):
        self.sentinel.write_text("changed")
        with self.assertRaises(RuntimeError):self.run_fake()
        self.assertFalse(FakeModel.calls)
    def test_control_fit_failure_stops(self):
        with patch.object(FakeModel,"fit",side_effect=RuntimeError("injected")):
            with self.assertRaises(RuntimeError):self.run_fake()
        diag=pd.read_csv(self.config.output_dir/"fold_diagnostics.csv")
        self.assertFalse(diag.loc[0,"valid"])
    def test_warning_from_ranker_propagates_to_subset(self):
        original=FakeModel.fit
        def warned(obj,x,y):
            if x.shape[1]==150:warnings.warn("ranker warning",UserWarning)
            return original(obj,x,y)
        with patch.object(FakeModel,"fit",warned):
            self.assertEqual(self.run_fake(),2)
        comparison=pd.read_csv(self.config.output_dir/"feature_selection_comparison.csv")
        self.assertFalse(comparison["eligible"].any())
        self.assertTrue(comparison["review_required"].all())
    def test_subset_failure_does_not_average_successful_folds(self):
        original=FakeModel.fit
        def fail_subset(obj,x,y):
            if x.shape[1]==5:raise RuntimeError("subset fail")
            return original(obj,x,y)
        with patch.object(FakeModel,"fit",fail_subset):self.assertEqual(self.run_fake(),2)
        comparison=pd.read_csv(self.config.output_dir/"feature_selection_comparison.csv")
        row=comparison.loc[comparison.k==5].iloc[0]
        self.assertFalse(row["complete_valid"])
        self.assertTrue(pd.isna(row["mean_auc"]))
    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None,"Real XGBoost unavailable; run this test locally.")
    def test_real_xgboost_control_and_selected_fold(self):
        from src.modeling.xgboost_pipeline_factory import XGBoostPipelineFactory
        factory=XGBoostPipelineFactory(overrides=dict(learning_rate=.08316404045417083,min_child_weight=1.229594255114693,
            reg_lambda=.1877052106059947,subsample=.9920641940615834,colsample_bytree=.7550128116345981))
        self.context["factory"]=factory
        self.context["parameters"]=factory.parameters(self.context["base"],4,50)
        split=self.context["splits"][0];tr,va=split["train_indices"],split["valid_indices"]
        x=self.context["x"]
        rec,_,model,_=self.runner._fit(self.context,split,150,"control",x.iloc[tr],x.iloc[va])
        self.assertTrue(rec["valid"],rec["error"])
        selector=GainTopKSelector(5).fit_from_booster(model.named_steps["classifier"].get_booster(),self.context["features"])
        rec,_,_,_=self.runner._fit(self.context,split,5,"selected",selector.transform(x.iloc[tr]),selector.transform(x.iloc[va]))
        self.assertTrue(rec["valid"],rec["error"])


if __name__=="__main__":unittest.main(verbosity=2)

"""Synthetic orchestration, leakage, callback-contract and integrity regression tests."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from scripts import run_ensemble_research_suite_v1 as m


def context(tmp_path):
    x,y=make_classification(n_samples=40,n_features=8,n_informative=4,random_state=7)
    x=pd.DataFrame(x,index=range(40));y=pd.Series(y)
    groups=[];metadata=[];matrices=[];offset=0
    rng=np.random.default_rng(42)
    for r in range(10):
        for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
            groups.append(dict(repeat=r,fold=f,split_seed=r,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            offset+=len(va)
            matrices.append(rng.uniform(.1,.9,(len(va),3)))
            metadata.extend(dict(repeat=r,fold=f,split_seed=r,row_index=int(i),id=f"00{i}",target=int(y.iloc[i])) for i in va)
    watched=tmp_path/"source.txt";watched.write_text("immutable")
    base=np.concatenate(matrices)
    return dict(context=dict(x=x,y=y,ids=pd.Series([f"00{i}" for i in range(40)]),candidate=dict(k=3,estimator_parameters=dict(n_estimators=50,random_state=42,n_jobs=1))),
                lightgbm_parameters=dict(n_estimators=50,num_leaves=7,n_jobs=1),groups=groups,frame=pd.DataFrame(metadata),
                base=base,baseline=base.mean(axis=1),output_root=tmp_path/"results",hashes={str(watched):m.digest(watched)},
                source_signature="frozen-signature",cohort="frozen-cohort",suite_versions={"synthetic":True},reproduction=[])


class NativeSurrogate:
    calls=[]
    def __init__(self,**params): self.params=params
    def fit(self,x,y,eval_set=None,eval_metric=None,callbacks=None,verbose=None):
        self.calls.append(set(x.index) if hasattr(x,"index") else None)
        self.model=LogisticRegression(C=.1,max_iter=1000).fit(x,y)
        self.classes_=self.model.classes_;self.best_iteration_=5;self.best_iteration=4;self.n_estimators_=5
        self.evals_result_={}; self.results={}
        if eval_set:
            vx,vy=eval_set[0];p=self.predict_proba(vx)[:,1]
            fn=eval_metric or self.params.get("eval_metric")
            result=fn(np.asarray(vy),p)
            if isinstance(result,tuple):
                name,value,maximize=result;assert maximize is True
                assert self.params.get("metric")=="None"
                self.evals_result_={"valid_0":{name:[value]}}
            else:
                assert callbacks is None
                cb=self.params["callbacks"][0]
                assert cb.maximize is True and cb.metric_name=="ensemble_auc"
                self.results={"validation_0":{"ensemble_auc":[result]}}
        return self
    def predict_proba(self,x): return self.model.predict_proba(x)
    def get_booster(self):
        return SimpleNamespace(num_boosted_rounds=lambda:self.params.get("n_estimators",50))
    def evals_result(self): return self.results


class Selector:
    def __init__(self,k): self.k=k
    def fit_from_booster(self,booster,features): self.features=features[:self.k];return self
    def transform(self,x): return x[self.features]


class BaseSurrogate:
    calls=[]
    def __init__(self,name): self.name=name
    def fit(self,x,y):
        self.calls.append((self.name,set(x.index)))
        if self.name=="xgboost":
            self.ranker=LogisticRegression(C=.1,max_iter=1000).fit(x,y)
        self.model=LogisticRegression(C=.1,max_iter=1000).fit(x,y)
        self.classes_=self.model.classes_
        return self
    def predict_proba(self,x): return self.model.predict_proba(x)


def adapters(monkeypatch):
    xgb=ModuleType("xgboost");xgb.XGBClassifier=NativeSurrogate
    callbacks=ModuleType("xgboost.callback")
    callbacks.EarlyStopping=lambda **kwargs:SimpleNamespace(**kwargs)
    xgb.callback=callbacks
    lgb=ModuleType("lightgbm");lgb.LGBMClassifier=NativeSurrogate
    def early_stopping(n,first_metric_only,verbose):
        assert first_metric_only is True
        return SimpleNamespace(patience=n)
    lgb.early_stopping=early_stopping
    cat=ModuleType("catboost");cat.CatBoostClassifier=NativeSurrogate
    feature=ModuleType("src.features.gain_top_k_selector");feature.GainTopKSelector=Selector
    for name,module in [("xgboost",xgb),("xgboost.callback",callbacks),("lightgbm",lgb),("catboost",cat),("src.features.gain_top_k_selector",feature)]:
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(m,"base_model",lambda c,name:BaseSurrogate(name))
    monkeypatch.setattr(m,"require_packages",lambda task:None)


def stats(): return dict(attempted_estimator_fits=0,completed_estimator_fits=0,warnings=[])


def test_frozen_plan_and_grids():
    assert len(m.cat_parameters())==16
    assert len(m.cat_parameters())*len(m.ALPHAS)==48
    assert m.plan()["total_planned_estimator_fits"]==2000
    assert m.WEIGHTS[0]==(1/3,1/3,1/3)


def test_full_six_task_orchestration(tmp_path,monkeypatch):
    adapters(monkeypatch);c=context(tmp_path)
    expected={"catboost":800,"weights":0,"lgbm_es":250,"xgb_es":300,"stacking":650}
    for task in m.TASKS:
        result=m.run_task(c,task)
        if task in expected:
            assert result["status"]=="completed"
            assert result["completed_estimator_fits"]==expected[task]
        else: assert result["status"] in ("completed","skipped")
        root=c["output_root"]/task
        assert m.read(root/"decision.json")["automatic_promotion"] is False
        for name,h in m.read(root/"run_manifest.json")["artifact_sha256"].items(): assert m.digest(root/name)==h
    comp=pd.read_csv(c["output_root"]/"catboost/comparison.csv")
    assert len(comp)==48
    assert len(pd.read_csv(c["output_root"]/"weights/comparison.csv"))==4
    with pytest.raises(FileExistsError): m.run_task(c,"weights")


@pytest.mark.parametrize("task",["weights","catboost","lgbm_es","xgb_es","stacking"])
def test_preflight_zero_fit_no_output(task,tmp_path,monkeypatch):
    adapters(monkeypatch);c=context(tmp_path)
    def forbidden(*args,**kwargs): raise AssertionError("Unexpected fit")
    monkeypatch.setattr(NativeSurrogate,"fit",forbidden)
    assert m.run_task(c,task,dry_run=True)["fits_performed"]==0
    assert not c["output_root"].exists()


@pytest.mark.parametrize("name",["lightgbm","xgboost"])
def test_es_inner_rows_do_not_include_outer_validation(name,tmp_path,monkeypatch):
    adapters(monkeypatch);c=context(tmp_path);s=c["groups"][0]
    BaseSurrogate.calls=[];NativeSurrogate.calls=[]
    count=stats();rounds,trace=m.select_rounds(c,s,name,count)
    assert rounds==5
    innertrain=set(trace["inner_train_indices"]);innervalid=set(trace["inner_valid_indices"])
    outervalid=set(s["valid_indices"])
    assert not innertrain&innervalid and not (innertrain|innervalid)&outervalid
    for _,rows in BaseSurrogate.calls: assert rows==innertrain
    for rows in NativeSurrogate.calls: assert rows==innertrain
    p,actual=m.refit_rounds(c,s,name,rounds,count)
    assert len(p)==len(s["valid_indices"])


def test_stacking_inner_oof_no_outer_validation(tmp_path,monkeypatch):
    adapters(monkeypatch);c=context(tmp_path);s=c["groups"][0]
    BaseSurrogate.calls=[]
    p,trace=m.stack_fold(c,s,stats())
    seen=[]
    for split in trace["inner_splits"]:
        a,b=set(split["train_indices"]),set(split["valid_indices"])
        assert not a&b and a|b==set(s["train_indices"])
        assert not (a|b)&set(s["valid_indices"])
        seen.extend(b)
    assert sorted(seen)==sorted(s["train_indices"])
    assert len(p)==len(s["valid_indices"])


def test_auc_metric_explicit_maximize():
    a=np.array([.1,.2,.8,.9]);y=np.array([0,0,1,1])
    name,value,maximize=m.metric_factory(a,a,"lightgbm")(y,a)
    assert name=="ensemble_auc" and value==1 and maximize is True
    assert m.metric_factory(a,a,"xgboost")(y,a)==1


def test_source_artifact_tamper_guard(tmp_path,monkeypatch):
    adapters(monkeypatch);c=context(tmp_path)
    m.run_task(c,"weights")
    root=c["output_root"]/"weights"
    with (root/"comparison.csv").open("a") as f:f.write("tamper")
    with pytest.raises(ValueError,match="artifact"):m.load_child(c,"weights")


def test_child_cohort_mismatch_guard(tmp_path,monkeypatch):
    adapters(monkeypatch);c=context(tmp_path);m.run_task(c,"weights")
    c["cohort"]="different"
    with pytest.raises(ValueError,match="cohort/design/reference"):m.load_child(c,"weights")


def test_reference_ties_not_wins(tmp_path):
    c=context(tmp_path)
    _,_,summary,_,comparison=m.score(c,{"reference":c["baseline"],"identical":c["baseline"]})
    d=m.decision_for(summary,comparison,{"identical":{}})
    assert d["ties"]==10 and d["wins"]==0 and not d["screening_passed"]


def test_probability_guard():
    for x in [np.array([np.nan]),np.array([1.1]),np.array([-.1])]:
        with pytest.raises(ValueError):m.probability(x,1)


def test_fit_convergence_warning_is_failure():
    class Bad:
        def fit(self,x,y):warnings.warn("test",ConvergenceWarning)
    count=stats()
    with pytest.raises(RuntimeError,match="ConvergenceWarning"):m.fit(Bad(),[],[],1,count)
    assert count["completed_estimator_fits"]==0


def test_failed_task_manifest_no_decision(tmp_path,monkeypatch):
    adapters(monkeypatch);c=context(tmp_path)
    def fail(*args):raise RuntimeError("deliberate")
    monkeypatch.setattr(m,"calculate",fail)
    with pytest.raises(RuntimeError):m.run_task(c,"weights")
    assert m.read(c["output_root"]/"weights/run_manifest.json")["status"]=="failed_or_interrupted"
    assert not (c["output_root"]/"weights/decision.json").exists()


def test_hash_guard(tmp_path):
    p=tmp_path/"x";p.write_text("first");hashes={str(p):m.digest(p)};p.write_text("second")
    with pytest.raises(ValueError):m.check_hashes(hashes)


def test_combination_skips_when_fewer_than_two_pass(tmp_path,monkeypatch):
    c=context(tmp_path)
    monkeypatch.setattr(m,"load_child",lambda c,t:({"screening_passed":False},{},{}))
    preds,specs,aux,traces,hashes=m.combine(c)
    assert not preds and "Fewer than two" in traces[0]["skipped"]


def test_combination_applies_only_passing_changes(tmp_path,monkeypatch):
    c=context(tmp_path);n=len(c["frame"])
    def child(c,task):
        if task=="catboost":return dict(screening_passed=True,selected_spec=dict(alpha=.1,component_key="cat_0")),{"component__cat_0":np.full(n,.7)},{}
        if task=="lgbm_es":return dict(screening_passed=True),{"component__lightgbm":np.full(n,.8)},{}
        return dict(screening_passed=False),{},{}
    monkeypatch.setattr(m,"load_child",child)
    preds,specs,*_=m.combine(c)
    base=c["base"].copy();base[:,2]=.8
    assert np.allclose(preds["all_passing_combined"],.9*base.mean(axis=1)+.1*.7)
    assert specs["all_passing_combined"]["included_changes"]==["catboost","lgbm_es"]


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None or importlib.util.find_spec("xgboost") is None,reason="Native LightGBM/XGBoost unavailable")
def test_native_early_stopping_callbacks(tmp_path,monkeypatch):
    # Real native constructors/callbacks with a tiny selector shim; project provenance is a separate local check.
    c=context(tmp_path)
    c["lightgbm_parameters"]=dict(n_estimators=10,num_leaves=3,min_child_samples=2,verbosity=-1,n_jobs=1)
    c["context"]["candidate"]["estimator_parameters"]=dict(n_estimators=10,max_depth=2,random_state=42,n_jobs=1)
    monkeypatch.setattr(m,"base_model",lambda c,name:BaseSurrogate(name))
    feature=ModuleType("src.features.gain_top_k_selector");feature.GainTopKSelector=Selector
    monkeypatch.setitem(sys.modules,"src.features.gain_top_k_selector",feature)
    for name in ("lightgbm","xgboost"):
        rounds,_=m.select_rounds(c,c["groups"][0],name,stats())
        assert 1<=rounds<=m.DESIGN["early_stop_cap"]


@pytest.mark.skipif(importlib.util.find_spec("catboost") is None,reason="Native CatBoost unavailable")
def test_native_catboost_smoke(tmp_path):
    c=context(tmp_path);params=m.cat_parameters()[0].copy();params["iterations"]=5
    p=m.cat_fold(c,c["groups"][0],params,stats())
    assert len(p)==8


def fixture_historical_source(tmp_path,monkeypatch):
    rng=np.random.default_rng(19)
    x=pd.DataFrame(rng.normal(size=(250,150)))
    y=pd.Series([0,1]*125);ids=pd.Series([f"000{i}" for i in range(250)],dtype="string")
    splits=[];rows=[];folds=[];reference=[];means=[]
    for r in range(10):
        for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
            splits.append(dict(scheme="cv5",repeat=r,split_seed=r,fold=f,train_indices=tr.tolist(),valid_indices=va.tolist()))
            matrix=rng.uniform(.1,.9,(len(va),3));avg=matrix.mean(axis=1)
            for i,(p,q) in enumerate(zip(matrix,avg)):
                row=int(va[i]);rows.append(dict(scheme="cv5",repeat=r,split_seed=r,fold=f,row_index=row,id=str(ids.iloc[row]),target=int(y.iloc[row]),l1_c1=p[0],xgboost=p[1],lightgbm=p[2],l1_xgb_lgbm=q))
            from sklearn.metrics import roc_auc_score
            tree=dict(repeat=r,fold=f,xgboost=float(roc_auc_score(y.iloc[va],matrix[:,1])),lightgbm=float(roc_auc_score(y.iloc[va],matrix[:,2])))
            reference.append(tree)
            for name,p in [("l1_c01",matrix[:,0]),("l1_c1",matrix[:,0]),("l1_xgb",matrix[:,:2].mean(axis=1)),("l1_xgb_lgbm",avg)]:
                auc=float(roc_auc_score(y.iloc[va],p))
                folds.append(dict(scheme="cv5",repeat=r,split_seed=r,fold=f,candidate=name,auc=auc))
                if name=="l1_xgb_lgbm":means.append(auc)
    source=tmp_path/"historical";source.mkdir()
    m.write(source/"split_indices.json",splits)
    pd.DataFrame(rows).to_csv(source/"validation_probabilities.csv",index=False)
    pd.DataFrame(folds).to_csv(source/"fold_scores.csv",index=False)
    m.write(source/"settings.json",{})
    m.write(source/"run_manifest.json",dict(status="completed",artifact_sha256={p.name:m.digest(p) for p in source.iterdir()}))
    val_cfg=tmp_path/"validation.json";m.write(val_cfg,{"output_dir":"historical"})
    m.write(tmp_path/"suite.json",dict(validation_config="validation.json",output_root="suite_outputs",design=m.DESIGN))
    fake=dict(context=dict(x=x,y=y,ids=ids),splits=splits,hashes={},reference_matches={(r,f):True for r in range(10) for f in range(5)},reference=pd.DataFrame(reference).set_index(["repeat","fold"]))
    class Verifier:
        def _prepare(self,*args):return fake
    module=ModuleType("src.extensions.validation_protocol_v1");module.ValidationProtocolV1=Verifier
    monkeypatch.setitem(sys.modules,"src.extensions.validation_protocol_v1",module)
    monkeypatch.setattr(m,"REFERENCE_AUC",float(np.mean(means)))
    return fake,source


def test_source_adapter_four_procedure_schema(tmp_path,monkeypatch):
    fixture_historical_source(tmp_path,monkeypatch)
    c=m.prepare("suite.json",root=tmp_path)
    assert c["base"].shape==(2500,3)
    assert len(c["reproduction"])==200
    assert not c["output_root"].exists()
    assert len(c["groups"])==50
    assert str(c["frame"].id.dtype)=="string"


def test_source_adapter_refuses_changed_training_order(tmp_path,monkeypatch):
    fake,_=fixture_historical_source(tmp_path,monkeypatch)
    fake["reference_matches"][(0,0)]=False
    with pytest.raises(ValueError,match="ordered split mismatch"):
        m.prepare("suite.json",root=tmp_path)


def test_source_adapter_refuses_tampered_csv(tmp_path,monkeypatch):
    _,source=fixture_historical_source(tmp_path,monkeypatch)
    with (source/"validation_probabilities.csv").open("a") as f:f.write("\n")
    with pytest.raises(ValueError,match="artifact"):
        m.prepare("suite.json",root=tmp_path)


def test_equal_weight_control_is_bitwise_reference(tmp_path):
    c=context(tmp_path)
    preds,*_=m.calculate("weights",c,stats())
    assert np.array_equal(preds["weights_0"],c["baseline"])

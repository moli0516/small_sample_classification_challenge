"""Optuna contracts and actual logistic-fit loop; native Optuna smoke is optional."""
import importlib.util
from pathlib import Path
from types import ModuleType,SimpleNamespace
import hashlib,json,sys
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from scripts import run_elasticnet_optuna_v1 as m


def contract():
    # Use real dependency when installed. Minimal contract otherwise, not native-source integration.
    if importlib.util.find_spec("scripts.run_elasticnet_ensemble_v1") is not None:
        from scripts import run_elasticnet_ensemble_v1
        return run_elasticnet_ensemble_v1
    e=ModuleType("elasticnet_test_contract");e.TOL=1e-12
    def safe(x):
        if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
        if isinstance(x,list):return [safe(v) for v in x]
        return x.item() if isinstance(x,np.generic) else x
    e.safe=safe;e.digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    e.write=lambda p,d:Path(p).write_text(json.dumps(safe(d),allow_nan=False))
    def check(hashes):
        for p,h in hashes.items():
            if e.digest(p)!=h:raise ValueError("changed input")
    e.check_hashes=check
    def prob(p,n):
        p=np.asarray(p,dtype=float)
        if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("invalid probability")
        return p
    e.probability=prob
    def build(spec):return Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(C=spec["C"],l1_ratio=spec["l1_ratio"],penalty="elasticnet",solver="saga",max_iter=20000,tol=1e-4,random_state=42))])
    e.build_model=build
    def score(c,preds):
        rows=[]
        for name,p in preds.items():
            for s in c["groups"]:
                sl=slice(s["start"],s["stop"])
                rows.append(dict(candidate=name,repeat=s["repeat"],fold=s["fold"],auc=float(roc_auc_score(c["frame"].target.iloc[sl],p[sl]))))
        f=pd.DataFrame(rows);r=f.groupby(["candidate","repeat"],as_index=False).auc.mean().rename(columns={"auc":"mean_fold_auc"})
        ss=r.groupby("candidate").mean_fold_auc.agg(mean_auc="mean",repeat_sd="std").reset_index()
        return f,r,ss
    e.score=score
    def compare(repeats,names,reference):
        get=lambda name:repeats[repeats.candidate==name].set_index("repeat").mean_fold_auc.sort_index()
        rows=[];details=[]
        for name in names:
            d=get(name)-get(reference)
            rows.append(dict(reference=reference,candidate=name,mean_difference=float(d.mean()),difference_sd=float(d.std(ddof=1)),wins=int((d>1e-12).sum()),losses=int((d< -1e-12).sum()),ties=int((abs(d)<=1e-12).sum())))
            details.extend(dict(repeat=int(r),candidate=name,difference=float(v)) for r,v in d.items())
        return pd.DataFrame(details),pd.DataFrame(rows)
    e.compare=compare
    return e


def context(tmp_path):
    x,y=make_classification(n_samples=40,n_features=8,n_informative=4,random_state=3)
    x=pd.DataFrame(x);y=pd.Series(y);groups=[];rows=[];offset=0
    for r in range(10):
        for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
            groups.append(dict(repeat=r,fold=f,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            rows.extend(dict(repeat=r,fold=f,row_index=int(i),target=int(y.iloc[i])) for i in va);offset+=len(va)
    p=tmp_path/"source";p.write_text("immutable")
    return dict(context=dict(x=x,y=y),groups=groups,frame=pd.DataFrame(rows),baseline=np.linspace(.1,.9,offset),
                catboost_control=np.linspace(.11,.89,offset),elasticnet_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()},
                output=tmp_path/"result",optuna_config=dict(design=m.DESIGN.copy()))


class Trial:
    def __init__(self,number,params):self.number=number;self.params=params.copy();self.user_attrs={};self.value=None;self.state=SimpleNamespace(name="RUNNING")
    def suggest_float(self,name,low,high,log=False):
        assert low<=self.params[name]<=high
        if name=="C":assert log is True
        else:assert log is False
        return self.params[name]
    def set_user_attr(self,k,v):self.user_attrs[k]=v


class Study:
    def __init__(self):self.trials=[];self.queued=[]
    def enqueue_trial(self,p):self.queued.append(p)
    def optimize(self,objective,n_trials,n_jobs,callbacks):
        assert n_jobs==1
        for i in range(n_trials):
            p=self.queued.pop(0) if self.queued else dict(C=.01+i*.01,l1_ratio=.85,alpha=.03)
            t=Trial(i,p);self.trials.append(t)
            try:t.value=objective(t);t.state=SimpleNamespace(name="COMPLETE")
            except BaseException:
                t.state=SimpleNamespace(name="FAIL")
                for callback in callbacks:callback(self,t)
                raise
            for callback in callbacks:callback(self,t)


def optuna_contract():
    def tpe(**kw):
        assert kw==dict(seed=42,n_startup_trials=10)
        return SimpleNamespace(**kw)
    def create(**kw):
        assert kw["direction"]=="maximize"
        return Study()
    return SimpleNamespace(__version__="synthetic-contract",samplers=SimpleNamespace(TPESampler=tpe),pruners=SimpleNamespace(NopPruner=lambda:object()),create_study=create)


def control_score(c,e):
    raw=np.empty(len(c["frame"]))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for s in c["groups"]:
            model=e.build_model(m.CONTROL).fit(c["context"]["x"].iloc[s["train_indices"]],c["context"]["y"].iloc[s["train_indices"]])
            raw[s["start"]:s["stop"]]=model.predict_proba(c["context"]["x"].iloc[s["valid_indices"]])[:,1]
    _,_,ss=e.score(c,{"candidate":.95*c["baseline"]+.05*raw})
    return float(ss.mean_auc.iloc[0])


def test_default_budget_bounds():
    assert m.DESIGN["n_trials"]==30
    assert m.DESIGN["n_trials"]*50==1500
    params=m.suggest(Trial(0,m.CONTROL))
    assert params==m.CONTROL


def test_complete_three_trial_study_with_real_logistic_fits(tmp_path,monkeypatch):
    c=context(tmp_path);e=contract()
    monkeypatch.setattr(m,"en_module",lambda:e)
    monkeypatch.setattr(m,"CONTROL_AUC",control_score(c,e))
    monkeypatch.setitem(m.DESIGN,"n_trials",3)
    result=m.execute(c,optuna_contract())
    assert result["completed_fits"]==150
    manifest=m.read(c["output"]/"run_manifest.json")
    assert manifest["status"]=="completed" and manifest["completed_trials"]==3
    decision=m.read(c["output"]/"decision.json")
    assert decision["control_reproduced"] is True and decision["automatic_promotion"] is False
    for name,h in manifest["artifact_sha256"].items():assert e.digest(c["output"]/name)==h
    assert len(pd.read_csv(c["output"]/"trials.csv"))==3
    with pytest.raises(FileExistsError):m.execute(c,optuna_contract())


def test_control_mismatch_stops_study(tmp_path,monkeypatch):
    c=context(tmp_path);e=contract()
    monkeypatch.setattr(m,"en_module",lambda:e)
    monkeypatch.setattr(m,"CONTROL_AUC",-1.)
    with pytest.raises(ValueError,match="Control failed"):
        m.execute(c,optuna_contract())
    assert m.read(c["output"]/"run_manifest.json")["status"]=="failed_or_interrupted"
    assert not (c["output"]/"decision.json").exists()
    assert len(m.read(c["output"]/"trials.json"))==1


def test_no_partial_selection(monkeypatch):
    t=Trial(0,m.CONTROL);t.value=.9;t.state=SimpleNamespace(name="COMPLETE")
    with pytest.raises(ValueError,match="All30"):
        m.select_best([t])


def test_ties_select_earliest(monkeypatch):
    monkeypatch.setitem(m.DESIGN,"n_trials",2)
    a,b=Trial(1,m.CONTROL),Trial(0,m.CONTROL)
    for t in [a,b]:t.value=.9;t.state=SimpleNamespace(name="COMPLETE")
    assert m.select_best([a,b]).number==0


def test_prepare_passes_new_destination_not_completed_grid_folder(tmp_path,monkeypatch):
    c=context(tmp_path);e=contract()
    (tmp_path/"old_grid_output").mkdir()
    seen=[]
    def prep(config,output):
        seen.append(Path(output));assert Path(output)!=tmp_path/"old_grid_output"
        return c
    e.prepare=prep
    monkeypatch.setattr(m,"en_module",lambda:e)
    monkeypatch.setattr(m,"ROOT",tmp_path)
    cfg=dict(suite_config="suite.json",output_dir="optuna_new",design=m.DESIGN.copy())
    (tmp_path/"config.json").write_text(json.dumps(cfg))
    prepared=m.prepare("config.json")
    assert seen==[tmp_path/"optuna_new"]
    assert prepared["output"]==tmp_path/"optuna_new" and not prepared["output"].exists()


def test_config_design_changes_rejected():
    cfg=dict(suite_config="x",output_dir="y",design=m.DESIGN.copy())
    cfg["design"]["n_trials"]=999
    with pytest.raises(ValueError,match="frozen"):m.validate(cfg)


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None,reason="Native Optuna unavailable")
def test_native_optuna_enqueue_sampler_contract():
    import optuna
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=10),pruner=optuna.pruners.NopPruner())
    study.enqueue_trial(m.CONTROL)
    study.optimize(lambda t:sum(m.suggest(t).values()),n_trials=1,n_jobs=1)
    assert study.trials[0].params==m.CONTROL

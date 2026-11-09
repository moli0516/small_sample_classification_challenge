"""Synthetic protocol tests; native selected-source preflight runs on the owner machine."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.extensions import validation_protocol_v1 as m


class Helper:
    @staticmethod
    def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @staticmethod
    def _write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False))
    @staticmethod
    def _probabilities(model,x):
        if not np.array_equal(model.classes_,[0,1]):raise ValueError("class order")
        return model.predict_proba(x)[:,1]


def cfg():
    return dict(source_study="study",sparse_l1_config="sparse.json",output_dir="run",
                split_seeds=list(range(10)),holdout_fraction=.2,k_values=[3,5,10])


@pytest.fixture
def data():
    rng=np.random.default_rng(7)
    return pd.DataFrame(rng.normal(size=(250,150))),pd.Series([0,1]*125)


def test_design_counts_sizes_and_no_overlap(data):
    x,y=data;splits=m.make_splits(y,cfg())
    assert len(splits)==190 and 5*len(splits)==950
    for s in splits:
        assert set(s["train_indices"]).isdisjoint(s["valid_indices"])
        if s["scheme"]=="holdout20":assert len(s["train_indices"])==200 and len(s["valid_indices"])==50
        if s["scheme"]=="cv10":assert len(s["train_indices"])==225 and len(s["valid_indices"])==25


def test_reproducible_split_seed_only(data):
    _,y=data
    a=m.make_splits(y,cfg());b=m.make_splits(y,cfg())
    assert a==b
    hold=[s for s in a if s["scheme"]=="holdout20"]
    assert hold[0]["valid_indices"]!=hold[1]["valid_indices"]


@pytest.mark.parametrize("field,value",[("split_seeds",[0,1]),("k_values",[5,10]),("holdout_fraction",.3),("split_seeds",[False]+list(range(1,10)))])
def test_config_rejects_changed_protocol(field,value):
    with pytest.raises(ValueError):m.validate(dict(cfg(),**{field:value}))


@pytest.mark.parametrize("mutation",["overlap","duplicate","seed","missing","class"])
def test_invalid_splits_rejected(data,mutation):
    _,y=data;s=m.make_splits(y,cfg())
    if mutation=="overlap":s[0]["train_indices"].append(s[0]["valid_indices"][0])
    if mutation=="duplicate":s[0]["valid_indices"].append(s[0]["valid_indices"][0])
    if mutation=="seed":s[0]["split_seed"]=99
    if mutation=="missing":s=s[:-1]
    if mutation=="class":y=pd.Series([0]*250)
    with pytest.raises(ValueError):m.validate_splits(s,y)


def results(data):
    _,y=data;rows=[]
    for s in m.make_splits(y,cfg()):
        key={k:s[k] for k in ("scheme","repeat","split_seed","fold")}
        base=.75+.001*s["repeat"]+.0001*s["fold"]
        for name,d in zip(m.MODELS,[0,.01,.03,.035]):
            rows.append(dict(key,candidate=name,auc=base+d,train_rows=len(s["train_indices"]),valid_rows=len(s["valid_indices"])))
    return pd.DataFrame(rows)


def test_repeat_level_aggregation_and_pairing(data):
    raw=results(data);r,summary,pairs,table=m.summarize(raw)
    assert (len(r),len(summary),len(pairs),len(table))==(160,16,80,8)
    assert np.allclose(table.loc[table.pair=="linear","mean_difference"],.01)
    assert np.allclose(table.loc[table.pair=="ensemble","mean_difference"],.005)
    assert (table.wins_b==10).all() and not table.winner_changes_across_splits.any()
    row=r.loc[(r.scheme=="cv3")&(r.repeat==0)&(r.candidate=="l1_c01")].iloc[0]
    assert np.isclose(row.mean_fold_auc,.7501)
    score=summary.loc[(summary.scheme=="holdout20")&(summary.candidate=="l1_c01")].iloc[0]
    assert np.isclose(score.repeat_sd,np.std(.75+.001*np.arange(10),ddof=1))


def test_rank_reversal_and_ties(data):
    raw=results(data)
    mask=(raw.candidate=="l1_c1")&(raw.repeat==0)
    raw.loc[mask,"auc"]-=.02
    mask2=(raw.candidate=="l1_c1")&(raw.repeat==1)
    raw.loc[mask2,"auc"]-=.01
    _,_,_,table=m.summarize(raw)
    rows=table.loc[table.pair=="linear"]
    assert (rows.wins_a==1).all() and (rows.ties==1).all() and rows.winner_changes_across_splits.all()


@pytest.mark.parametrize("mutation",["partial","nan","duplicate","size","seed"])
def test_invalid_results_not_scored(data,mutation):
    f=results(data)
    if mutation=="partial":f=f.iloc[:-1]
    if mutation=="nan":f.loc[0,"auc"]=np.nan
    if mutation=="duplicate":f=pd.concat([f.iloc[[0]],f.iloc[:-1]],ignore_index=True)
    if mutation=="size":f.loc[0,"train_rows"]+=1
    if mutation=="seed":f.loc[0,"split_seed"]=99
    with pytest.raises(ValueError):m.summarize(f)


def test_plots_created(data,tmp_path):
    r,_,p,_=m.summarize(results(data))
    paths=m.plot_results(r,p,tmp_path)
    assert len(paths)==2 and all(path.stat().st_size>0 for path in paths)


def prepared(data,tmp_path):
    x,y=data
    path=tmp_path/"input";path.write_text("original")
    c=dict(helper=Helper(),config=cfg(),output=tmp_path/"out",versions={},hashes={str(path):Helper._sha(path)},
        context=dict(x=x,y=y,ids=pd.Series([f"{i:04d}" for i in range(250)]),
            linear=SimpleNamespace(to_dict=lambda:{}),candidate=dict(estimator_parameters={})),
        sparse=SimpleNamespace(to_dict=lambda:{}),lightgbm_parameters={},splits=m.make_splits(y,cfg()),
        reference_matches={(r,f):False for r in range(10) for f in range(5)})
    c["reference"]=pd.DataFrame([dict(repeat=r,fold=f,l1=.5,xgboost=.5,lightgbm=.5,l1_xgb=.5,l1_xgb_lgbm=.5)
        for r in range(10) for f in range(5)]).set_index(["repeat","fold"])
    return c


def models_factory(records):
    class Model:
        def __init__(self,alias):self.alias=alias
        def fit(self,x,y):
            self.training=set(x.index);self.classes_=np.array([0,1]);records.append((self.alias,tuple(x.index)))
        def predict_proba(self,x):
            assert self.training.isdisjoint(x.index)
            p=1/(1+np.exp(-x.iloc[:,0].to_numpy()))
            return np.column_stack([1-p,p])
        def diagnostics(self):return dict(feature_count=150)
    return lambda c:[(name,Model(name),cost) for name,cost in [("l1_c01",1),("l1_c1",1),("xgboost",2),("lightgbm",1)]]


def test_mocked_full_lifecycle(data,tmp_path,monkeypatch,capsys):
    c=prepared(data,tmp_path);records=[];runner=m.ValidationProtocolV1()
    monkeypatch.setattr(runner,"_prepare",lambda *a:c);monkeypatch.setattr(runner,"_models",models_factory(records))
    dry=runner.run("unused",dry_run=True)
    assert dry["planned_estimator_fits"]==950 and not c["output"].exists() and not records
    runner.run("unused")
    assert len(records)==760 and "[190/190]" in capsys.readouterr().out
    for i in range(0,len(records),4):assert len({indices for _,indices in records[i:i+4]})==1
    out=c["output"];manifest=m.read(out/"run_manifest.json")
    assert manifest["status"]=="completed" and manifest["completed_estimator_fits"]==950 and manifest["completed_splits"]==190
    assert not manifest["test_opened"] and not manifest["kaggle_upload_performed"]
    assert len(pd.read_csv(out/"fold_scores.csv"))==760
    assert len(pd.read_csv(out/"validation_probabilities.csv"))==8000
    assert len(pd.read_csv(out/"paired_summary.csv"))==8
    assert not m.read(out/"interpretation.json")["automatic_protocol_selection"]
    for name,digest in manifest["artifact_sha256"].items():assert Helper._sha(out/name)==digest
    with pytest.raises(FileExistsError):runner.run("unused")


def test_identical_cv5_must_reproduce(data,tmp_path,monkeypatch):
    c=prepared(data,tmp_path);records=[];runner=m.ValidationProtocolV1()
    # Fail at the first split to keep the regression small; exercised as cv5.
    c["splits"]=[dict(c["splits"][0],scheme="cv5")]
    c["reference_matches"][(0,0)]=True
    monkeypatch.setattr(runner,"_prepare",lambda *a:c);monkeypatch.setattr(runner,"_models",models_factory(records))
    with pytest.raises(RuntimeError,match="reproduction"):runner.run("unused")
    assert m.read(c["output"]/"run_manifest.json")["status"]=="failed_or_interrupted"
    assert not (c["output"]/"interpretation.json").exists()


def test_warnings_stop_entire_study(data,tmp_path,monkeypatch):
    c=prepared(data,tmp_path);runner=m.ValidationProtocolV1()
    class Bad:
        def fit(self,x,y):
            warnings.warn("review warning",UserWarning)
            raise RuntimeError("fit failed")
    monkeypatch.setattr(runner,"_prepare",lambda *a:c);monkeypatch.setattr(runner,"_models",lambda c:[("l1_c01",Bad(),1)])
    with pytest.raises(RuntimeError):runner.run("unused")
    manifest=m.read(c["output"]/"run_manifest.json")
    assert manifest["completed_splits"]==0 and manifest["status"]=="failed_or_interrupted"
    assert m.read(c["output"]/"warnings.json")[0]["message"]=="review warning"


def test_changed_input_blocks_dry_run(data,tmp_path,monkeypatch):
    c=prepared(data,tmp_path);(tmp_path/"input").write_text("changed")
    runner=m.ValidationProtocolV1();monkeypatch.setattr(runner,"_prepare",lambda *a:c)
    with pytest.raises(RuntimeError):runner.run("unused",dry_run=True)
    assert not c["output"].exists()


def test_manifest_hash_and_path_guards(tmp_path):
    p=tmp_path/"settings.json";p.write_text("{}")
    Helper._write(tmp_path/"run_manifest.json",dict(status="completed",artifact_sha256={p.name:Helper._sha(p)}))
    m.verify_artifacts(tmp_path,Helper(),{"settings.json"})
    p.write_text("changed")
    with pytest.raises(ValueError):m.verify_artifacts(tmp_path,Helper(),{"settings.json"})
    Helper._write(tmp_path/"run_manifest.json",dict(status="completed",artifact_sha256={"../escape":"bad"}))
    with pytest.raises(ValueError):m.verify_artifacts(tmp_path,Helper(),set())


def test_real_pipeline_scaler_training_only(data):
    x,y=data;tr=list(range(200));va=list(range(200,250))
    altered=x.copy();altered.iloc[va,:]=1e6
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=42))
    model.fit(altered.iloc[tr],y.iloc[tr])
    assert np.allclose(model.named_steps["standardscaler"].mean_,x.iloc[tr].mean())
    assert np.isfinite(model.predict_proba(altered.iloc[va])).all()

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.modeling.semi_supervised_mixture_v1 import SemiSupervisedMixture
from src.features.generative_geometry_v1 import GeometryProjector
from src.experiments import generative_geometry_v1 as e


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):yield


@pytest.mark.parametrize("covariance",["diag","tied"])
def test_actual_em_probabilities_and_label_mapping(covariance):
    rng=np.random.RandomState(12)
    x=np.vstack([rng.normal(-3,.3,(25,3)),rng.normal(3,.3,(25,3))]);y=np.r_[np.zeros(25,int),np.ones(25,int)]
    model=SemiSupervisedMixture(2,covariance,random_state=42).fit(x,y,x+.02)
    p=model.predict_proba(x)
    assert model.converged_ and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1),1)
    assert roc_auc_score(y,p[:,1])>.99
    flipped=SemiSupervisedMixture(2,covariance,random_state=42).fit(x,1-y,x+.02)
    np.testing.assert_allclose(p[:,1],1-flipped.predict_proba(x)[:,1],atol=1e-7)
    assert np.all((model.class_probabilities_>0)&(model.class_probabilities_<1))


def test_reproducible_seed_and_normalized_objective():
    rng=np.random.RandomState(2);x=rng.normal(size=(40,3));y=np.tile([0,1],20);u=rng.normal(size=(70,3))
    a=SemiSupervisedMixture(2,unlabelled_weight=.4).fit(x,y,u)
    b=SemiSupervisedMixture(2,unlabelled_weight=.4).fit(x,y,u)
    np.testing.assert_allclose(a.predict_proba(u),b.predict_proba(u),atol=0,rtol=0)
    _,norms=a._terms(np.vstack([x,u]),y,len(x))
    expected=norms[:40].mean()+.4*norms[40:].mean()+a.label_smoothing*(np.log(a.class_probabilities_)+np.log1p(-a.class_probabilities_)).sum()
    assert a.objective_trace_[-1]==pytest.approx(expected)


def test_geometry_fit_scope_and_constant_columns():
    x=np.random.RandomState(4).normal(size=(40,8));x[:,0]=1
    p=GeometryProjector(5,3).fit(x[:30])
    before=p.scaler_.mean_.copy();z=p.transform(x[30:]+100)
    assert z.shape==(10,3) and 0 not in p.selected_indices_
    np.testing.assert_array_equal(before,p.scaler_.mean_)
    assert p.diagnostics_["rows_fitted"]==30
    with pytest.raises(ValueError):GeometryProjector().fit(np.ones((10,3)))


def test_model_rejects_invalid_inputs():
    x=np.zeros((10,2));y=np.tile([0,1],5)
    with pytest.raises(ValueError):SemiSupervisedMixture().fit(x,np.zeros(10))
    with pytest.raises(ValueError):SemiSupervisedMixture().fit(x,y,np.zeros((3,4)))
    with pytest.raises(ValueError):SemiSupervisedMixture(covariance_type="full").fit(x,y)


def miniature(tmp_path):
    x,y=make_classification(n_samples=40,n_features=8,n_informative=4,random_state=5)
    x=pd.DataFrame(x,columns=[f"f{i}" for i in range(8)]);y=pd.Series(y);ids=pd.Series([f"{i:04d}" for i in range(40)],dtype="string")
    groups=[];rows=[];offset=0
    for r in range(10):
        for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
            groups.append(dict(repeat=r,fold=f,split_seed=r,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            rows.extend(dict(repeat=r,fold=f,split_seed=r,row_index=int(i),id=str(ids.iloc[i]),target=int(y.iloc[i])) for i in va);offset+=len(va)
    frame=pd.DataFrame(rows);reference=.1+.8*frame.target.to_numpy()
    return dict(x=x,y=y,ids=ids,features=list(x.columns),groups=groups,frame=frame,reference=reference,reference_auc=1.,
        hashes={},config={},modes=["A","B"],output=tmp_path/"result")


def test_end_to_end_A_B_real_model_fits_and_artifacts(tmp_path,monkeypatch):
    c=miniature(tmp_path)
    # Exercise4 folds and two seeds; production preflight still requires all50 saved folds.
    selected=[c["groups"][i] for i in [0,1,5,6]]
    parts=[];groups=[];offset=0
    for s in selected:
        part=c["frame"].iloc[s["start"]:s["stop"]].copy();parts.append(part)
        groups.append(dict(s,start=offset,stop=offset+len(part)));offset+=len(part)
    c["groups"]=groups;c["frame"]=pd.concat(parts,ignore_index=True)
    c["reference"]=.1+.8*c["frame"].target.to_numpy()
    for k,v in dict(clusters=[2],covariances=["diag"],initialization_seeds=[42,314],max_iter=100,tol=.001).items():monkeypatch.setitem(e.DESIGN,k,v)
    decision=e.execute(c)
    manifest=e.read(c["output"]/"run_manifest.json")
    assert manifest["status"]=="completed" and manifest["completed_candidate_fits"]==16
    assert manifest["competition_test_opened"] is False
    assert decision["automatic_promotion"] is False
    assert len(pd.read_csv(c["output"]/"B_vs_A.csv"))==1
    geometry=e.read(c["output"]/"geometry_diagnostics.json")
    assert {r["rows_fitted"] for r in geometry if r["mode"]=="A"}=={32}
    assert {r["rows_fitted"] for r in geometry if r["mode"]=="B"}=={40}
    for name,h in manifest["artifact_sha256"].items():assert e.digest(c["output"]/name)==h
    with pytest.raises(FileExistsError):e.execute(c)


def bundle(tmp_path):
    c=miniature(tmp_path);source=tmp_path/"reference";source.mkdir()
    train=tmp_path/"train.csv";data=c["x"].copy();data.insert(0,"id",c["ids"]);data["target"]=c["y"];data.to_csv(train,index=False)
    e.write(source/"settings.json",dict(input_and_code_sha256={str(train):e.digest(train)}))
    e.write(source/"decision.json",dict(selected_trial=25,selected_mean_auc=1.,control_reproduced=True))
    e.write(source/"split_indices.json",c["groups"]);c["frame"].to_csv(source/"cohort.csv",index=False)
    p=source/"trials/trial_0025";p.mkdir(parents=True)
    np.savez_compressed(p/"predictions.npz",ensemble=c["reference"])
    hashes={str(p.relative_to(source)):e.digest(p) for p in source.rglob("*") if p.is_file()}
    e.write(source/"run_manifest.json",dict(status="completed",completed_trials=30,artifact_sha256=hashes))
    cfg=dict(train_path=str(train),target_column="target",id_column="id",expected_rows=40,expected_features=8,reference_study=str(source),output_dir=str(tmp_path/"new"),design=e.DESIGN.copy())
    cfgpath=tmp_path/"config.json";e.write(cfgpath,cfg)
    return c,cfgpath,source


def test_preflight_real_loader_no_test_open_no_writes(tmp_path,monkeypatch):
    c,path,source=bundle(tmp_path)
    import builtins
    original=builtins.open
    def guard(file,*args,**kwargs):
        if Path(file).name=="test.csv":raise AssertionError("Competition test touched")
        return original(file,*args,**kwargs)
    monkeypatch.setattr(builtins,"open",guard)
    prepared=e.prepare(str(path))
    assert prepared["reference_auc"]==1. and prepared["modes"]==["A"]
    assert not prepared["output"].exists()
    assert e.prepare(str(path),transductive_validation=True)["modes"]==["A","B"]


def test_reference_tampering_rejected(tmp_path):
    c,path,source=bundle(tmp_path)
    (source/"cohort.csv").write_text("tampered")
    with pytest.raises(ValueError,match="changed reference"):e.prepare(str(path))


def test_test_mode_unknown_fields_rejected_before_loading(tmp_path):
    cfg=dict(train_path="missing_train.csv",target_column="target",id_column="id",expected_rows=250,expected_features=150,
        reference_study="missing_ref",output_dir="missing_out",design=e.DESIGN.copy(),test_path="test.csv")
    p=tmp_path/"bad.json";e.write(p,cfg)
    with pytest.raises(ValueError,match="Competition-test mode"):e.prepare(str(p))


def test_reference_label_or_id_mismatch_rejected(tmp_path):
    c,path,source=bundle(tmp_path)
    frame=pd.read_csv(source/"cohort.csv",dtype={"id":"string"});frame.loc[0,"id"]="wrong";frame.to_csv(source/"cohort.csv",index=False)
    m=e.read(source/"run_manifest.json");m["artifact_sha256"]["cohort.csv"]=e.digest(source/"cohort.csv");e.write(source/"run_manifest.json",m)
    with pytest.raises(ValueError,match="identity mismatch"):e.prepare(str(path))


def test_cli_dry_run_budget_and_no_output(tmp_path):
    import subprocess,sys
    c,path,source=bundle(tmp_path)
    run=subprocess.run([sys.executable,"-m","scripts.run_generative_geometry_v1","--config",str(path),"--dry-run","--transductive-validation"],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stderr
    data=json.loads(run.stdout)
    assert data["candidate_fits"]==1800 and data["fits_performed"]==0
    assert data["modes"]==["A","B"] and data["competition_test_opened"] is False
    assert not Path(data["output"]).exists()

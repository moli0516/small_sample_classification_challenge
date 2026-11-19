import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments import generative_optuna_v1 as r
from src.experiments import generative_optuna_snapshot_v1 as s
from src.features.generative_geometry_v1 import GeometryProjector
from src.features.generative_search_geometry_v1 import SearchGeometryProjector


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):yield


class Trial:
    def __init__(self,number,params):self.number=number;self.params=params.copy();self.user_attrs={};self.state=SimpleNamespace(name="RUNNING");self.value=None
    def suggest_categorical(self,name,choices):
        assert self.params[name] in choices
        return self.params[name]
    def suggest_float(self,name,low,high,log=False):
        assert log and low<=self.params[name]<=high
        return self.params[name]
    def set_user_attr(self,k,v):self.user_attrs[k]=v


class Study:
    def __init__(self):self.trials=[];self.queued=[]
    def enqueue_trial(self,params):self.queued.append(params.copy())
    def optimize(self,objective,n_trials,n_jobs,catch,callbacks):
        assert n_jobs==1 and catch==(r.RejectedTrial,)
        for i in range(n_trials):
            t=Trial(i,self.queued.pop(0) if self.queued else r.CONTROL);self.trials.append(t)
            try:t.value=objective(t);t.state=SimpleNamespace(name="COMPLETE")
            except BaseException as exc:
                t.state=SimpleNamespace(name="FAIL")
                for callback in callbacks:callback(self,t)
                if not isinstance(exc,catch):raise
                continue
            for callback in callbacks:callback(self,t)


def fake_optuna():
    def sampler(**kwargs):
        assert kwargs==dict(seed=42,n_startup_trials=10)
        return kwargs
    def create(**kwargs):assert kwargs["direction"]=="maximize";return Study()
    return SimpleNamespace(__version__="API-test-double-not-TPE",create_study=create,samplers=SimpleNamespace(TPESampler=sampler),pruners=SimpleNamespace(NopPruner=lambda:object()))


def toy(tmp_path,monkeypatch):
    x,y=make_classification(n_samples=48,n_features=10,n_informative=5,n_redundant=0,random_state=11)
    x=pd.DataFrame(x);y=pd.Series(y);rows=[];groups=[];offset=0
    for repeat in range(2):
        for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=repeat).split(x,y)):
            if fold==2:continue
            groups.append(dict(repeat=repeat,fold=fold,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            rows.extend(dict(repeat=repeat,fold=fold,row_index=int(i),target=int(y.iloc[i]),id=str(i)) for i in va);offset+=len(va)
    c=dict(x=x,y=y,groups=groups,frame=pd.DataFrame(rows),hashes={},output=tmp_path/"result",optuna_config={})
    raw=np.empty(len(c["frame"]))
    # Independently use OLD projector API to build the synthetic frozen reference.
    for split in groups:
        tr,va=split["train_indices"],split["valid_indices"]
        p=GeometryProjector(20,8).fit(x.iloc[tr]);pred=[]
        for seed in r.b.DESIGN["initialization_seeds"]:
            m=r.model(r.CONTROL,seed).fit(p.transform(x.iloc[tr]),y.iloc[tr].to_numpy());assert m.converged_
            pred.append(m.predict_proba(p.transform(x.iloc[va]))[:,1])
        raw[split["start"]:split["stop"]]=np.mean(pred,axis=0)
    c["control"]=raw;c["reference"]=.2+.5*c["frame"].target.to_numpy();c["hybrid_control"]=.5*raw+.5*c["reference"]
    _,_,summary=r.b.evaluate(c,{"gmm":raw,"hybrid":c["hybrid_control"]});c["source_reproduction"]=summary
    means=summary.set_index("candidate").mean_auc
    monkeypatch.setattr(s,"GMM_AUC",float(means["gmm"]));monkeypatch.setattr(s,"HYBRID_AUC",float(means["hybrid"]))
    monkeypatch.setitem(s.DESIGN,"n_trials",3)
    return c


def test_default_budget_and_search_space():
    assert s.DESIGN["n_trials"]==30 and s.DESIGN["n_trials"]*50*3==4500
    assert r.suggest(Trial(0,r.CONTROL))==r.CONTROL


def test_geometry_preserves_old_arithmetic_and_clips_dimensions():
    x=np.random.RandomState(4).normal(size=(30,25))
    old=GeometryProjector(20,8).fit(x);new=SearchGeometryProjector(20,8).fit(x)
    np.testing.assert_array_equal(old.transform(x),new.transform(x))
    clip=SearchGeometryProjector(10,16).fit(x)
    assert clip.search_diagnostics_["effective_dimensions"]==10
    mean=clip.scaler_.mean_.copy();clip.transform(x+100);np.testing.assert_array_equal(mean,clip.scaler_.mean_)


def test_three_trials_real_native_models_and_control(tmp_path,monkeypatch):
    c=toy(tmp_path,monkeypatch);decision=r.execute(c,fake_optuna())
    m=r.b.read(c["output"]/"run_manifest.json")
    assert m["completed_trials"]==3 and m["attempted_trials"]==3 and m["completed_candidate_fits"]==36
    assert m["status"]=="completed" and m["rejected_trials"]==0 and m["competition_test_opened"] is False
    assert decision["selected_trial"]==0 and decision["screening_passed"] is None
    assert decision["screening_status"]=="not_applicable_control_selected" and decision["automatic_promotion"] is False
    for name,h in m["artifact_sha256"].items():assert r.b.digest(c["output"]/name)==h
    with pytest.raises(FileExistsError):r.execute(c,fake_optuna())


def test_control_row_prediction_mismatch_stops_before_trial1(tmp_path,monkeypatch):
    c=toy(tmp_path,monkeypatch);c["control"]=np.clip(c["control"]+.001,0,1)
    with pytest.raises(r.ControlFailure,match="row-level"):r.execute(c,fake_optuna())
    m=r.b.read(c["output"]/"run_manifest.json")
    assert m["attempted_trials"]==1 and m["status"]=="failed_or_interrupted"
    assert not (c["output"]/"decision.json").exists()


def test_rejection_counts_in_budget_without_replacement(tmp_path,monkeypatch):
    c=toy(tmp_path,monkeypatch);fit=r.fit_trial
    def wrapped(c,trial,params,stats):
        if trial.number==1:raise r.RejectedTrial("explicit synthetic nonconvergence")
        return fit(c,trial,params,stats)
    monkeypatch.setattr(r,"fit_trial",wrapped);r.execute(c,fake_optuna())
    m=r.b.read(c["output"]/"run_manifest.json")
    assert m["attempted_trials"]==3 and m["completed_trials"]==2 and m["rejected_trials"]==1 and m["completed_candidate_fits"]==24


def test_actual_nonconvergence_rejection_and_diagnostics(tmp_path,monkeypatch):
    c=toy(tmp_path,monkeypatch);build=r.model
    def model(params,seed):
        m=build(params,seed);fit=m.fit
        def forced(*a,**kw):value=fit(*a,**kw);m.converged_=False;return value
        m.fit=forced;return m
    monkeypatch.setattr(r,"model",model)
    with pytest.raises(r.RejectedTrial):r.fit_trial(c,Trial(1,r.CONTROL),r.CONTROL,dict(attempted_projector_fits=0,completed_projector_fits=0,attempted_candidate_fits=0,completed_candidate_fits=0,converged_candidate_fits=0,planned_candidate_fits=36))


def test_unknown_value_error_is_fatal_not_rejected(tmp_path,monkeypatch):
    c=toy(tmp_path,monkeypatch);fit=r.fit_trial
    def wrapped(c,trial,params,stats):
        if trial.number==1:raise ValueError("unexpected feature schema/programming error")
        return fit(c,trial,params,stats)
    monkeypatch.setattr(r,"fit_trial",wrapped)
    with pytest.raises(ValueError,match="schema/programming"):r.execute(c,fake_optuna())
    m=r.b.read(c["output"]/"run_manifest.json")
    assert m["attempted_trials"]==2 and m["rejected_trials"]==0 and not (c["output"]/"decision.json").exists()


def test_control_numerical_failure_is_fatal(tmp_path,monkeypatch):
    c=toy(tmp_path,monkeypatch)
    def bad(*a,**kw):raise np.linalg.LinAlgError("singular")
    monkeypatch.setattr(r,"fit_trial",bad)
    with pytest.raises(r.ControlFailure):r.execute(c,fake_optuna())
    assert r.b.read(c["output"]/"run_manifest.json")["attempted_trials"]==1


def test_earliest_exact_tie_and_no_screening_fallback():
    trials=[Trial(i,r.CONTROL) for i in range(3)]
    for t,value in zip(trials,[.8,.83,.82]):t.value=value;t.state=SimpleNamespace(name="COMPLETE")
    assert r.choose(SimpleNamespace(trials=trials)).number==1
    trials[2].value=.83;assert r.choose(SimpleNamespace(trials=trials)).number==1


def source(tmp_path,monkeypatch):
    root=tmp_path/"source";root.mkdir();y=np.tile([0,1],125);rows=[];groups=[];offset=0
    for repeat in range(10):
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=repeat).split(np.zeros((250,1)),y)):
            groups.append(dict(repeat=repeat,fold=fold,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            rows.extend(dict(repeat=repeat,fold=fold,row_index=int(i),id=f"{i:04d}",target=int(y[i])) for i in va);offset+=len(va)
    frame=pd.DataFrame(rows);frame["id"]=frame.id.astype("string");rng=np.random.RandomState(25);signal=frame.target.to_numpy()*2-1
    g=expit(signal+rng.normal(size=len(frame)));t=expit(.95*signal+rng.normal(size=len(frame)))
    c=dict(x=pd.DataFrame(np.zeros((250,150))),y=pd.Series(y),frame=frame,groups=groups,control=g,reference=t,hashes={},config={},followup_config={"fixture":True},source_study=str(tmp_path/"original"))
    arrays={"A_k8_tied":g,"reference_trial25":t,**{f"blend_gmm{w:g}":w*g+(1-w)*t for w in [.25,.5,.75]}}
    _,rr,ss=r.b.evaluate(c,arrays);means=ss.set_index("candidate").mean_auc
    monkeypatch.setattr(s,"GMM_AUC",float(means["A_k8_tied"]));monkeypatch.setattr(s,"HYBRID_AUC",float(means["blend_gmm0.5"]))
    comparisons=[r.b.compare(rr,"blend_gmm0.5",n) for n in ["A_k8_tied","reference_trial25"]]
    r.b.write(root/"settings.json",dict(config=c["followup_config"],source_design=r.b.DESIGN,source_study=c["source_study"],versions={},input_and_code_sha256={}))
    r.b.write(root/"decision.json",dict(selected_candidate="blend_gmm0.5",selected_spec={"gmm_weight":.5,"trial25_weight":.5},screening_against_both_controls=True,selected_mean_auc=float(means["blend_gmm0.5"]),comparisons=comparisons))
    frame.to_csv(root/"cohort.csv",index=False);r.b.write(root/"split_indices.json",groups);r.b.write(root/"candidate_specs.json",{});np.savez_compressed(root/"predictions.npz",**arrays)
    def rehash():
        r.b.write(root/"run_manifest.json",dict(status="completed",action="blend",competition_test_opened=False,completed_candidate_fits=0,planned_candidate_fits=0,artifact_sha256={p.name:r.b.digest(p) for p in root.iterdir() if p.name!="run_manifest.json"}))
    rehash();cfgpath=tmp_path/"cfg.json";r.b.write(cfgpath,dict(followup_config="fixture_followup.json",blend_study=str(root),output_dir=str(tmp_path/"new"),design=s.DESIGN.copy()))
    monkeypatch.setattr(s.frozen,"prepare",lambda config,action,output:dict(c,output=Path(output)))
    return cfgpath,root,rehash


def test_snapshot_full50_fold_source_no_write(tmp_path,monkeypatch):
    path,root,rehash=source(tmp_path,monkeypatch);c=s.prepare(str(path))
    assert len(c["groups"])==50 and not c["output"].exists()
    assert np.allclose(c["hybrid_control"],.5*c["control"]+.5*c["reference"])


def test_snapshot_corruption_rejected(tmp_path,monkeypatch):
    path,root,rehash=source(tmp_path,monkeypatch);(root/"cohort.csv").write_text("tampered")
    with pytest.raises(ValueError):s.prepare(str(path))


def test_snapshot_rehashed_identity_change_rejected(tmp_path,monkeypatch):
    path,root,rehash=source(tmp_path,monkeypatch);frame=pd.read_csv(root/"cohort.csv",dtype={"id":"string"});frame.loc[0,"id"]="wrong";frame.to_csv(root/"cohort.csv",index=False);rehash()
    with pytest.raises(ValueError,match="cohort/split"):s.prepare(str(path))


def test_snapshot_test_mode_refused_before_legacy_prepare(tmp_path):
    p=tmp_path/"bad.json";r.b.write(p,dict(followup_config="missing",blend_study="missing",output_dir="missing",design=s.DESIGN.copy(),test_path="test.csv"))
    with pytest.raises(ValueError,match="Frozen"):s.prepare(str(p))


def test_dry_run_format_zero_fits_no_output(tmp_path,monkeypatch,capsys):
    import sys
    c=dict(output=tmp_path/"new")
    monkeypatch.setattr(s,"prepare",lambda *a:c)
    monkeypatch.setitem(sys.modules,"optuna",fake_optuna())
    monkeypatch.setattr(sys,"argv",["generative-optuna","--dry-run"])
    r.main();value=json.loads(capsys.readouterr().out)
    assert value["fits_performed"]==0 and value["competition_test_opened"] is False and not c["output"].exists()


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None,reason="Native Optuna not installed in artifact-generation environment")
def test_native_optuna_end_to_end(tmp_path,monkeypatch):
    import optuna
    c=toy(tmp_path,monkeypatch);monkeypatch.setitem(s.DESIGN,"n_trials",2)
    r.execute(c,optuna)
    m=r.b.read(c["output"]/"run_manifest.json")
    assert m["status"]=="completed" and m["attempted_trials"]==2


@pytest.mark.parametrize("error",[np.linalg.LinAlgError("singular"),FloatingPointError("overflow"),ValueError("ill-defined empirical covariance")])
def test_only_declared_numerical_errors_are_rejected(tmp_path,monkeypatch,error):
    c=toy(tmp_path,monkeypatch);fit=r.fit_trial
    def wrapped(c,trial,params,stats):
        if trial.number==1:raise error
        return fit(c,trial,params,stats)
    monkeypatch.setattr(r,"fit_trial",wrapped);r.execute(c,fake_optuna())
    m=r.b.read(c["output"]/"run_manifest.json")
    assert m["attempted_trials"]==3 and m["rejected_trials"]==1 and m["status"]=="completed"


def test_hash_conflicts_are_fatal():
    with pytest.raises(ValueError,match="Conflicting"):s.merge_hashes({"x":"a"},{"x":"b"})


def test_changed_source_between_trials_aborts(tmp_path,monkeypatch):
    c=toy(tmp_path,monkeypatch);marker=tmp_path/"watched";marker.write_text("original");c["hashes"]={str(marker):r.b.digest(marker)};fit=r.fit_trial
    def wrapped(c,trial,params,stats):
        value=fit(c,trial,params,stats)
        if trial.number==0:marker.write_text("changed")
        return value
    monkeypatch.setattr(r,"fit_trial",wrapped)
    with pytest.raises(ValueError,match="Changed source"):r.execute(c,fake_optuna())
    assert not (c["output"]/"decision.json").exists()

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments import generative_followup_v1 as f
from src.submission import generative_export_v1 as ex
from src.features.generative_followup_v1 import PlainPCAProjector


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):yield


def context(tmp_path,action):
    x,y=make_classification(n_samples=48,n_features=6,n_informative=4,n_redundant=0,random_state=12)
    x=pd.DataFrame(x,columns=[f"f{i}" for i in range(6)]);y=pd.Series(y)
    groups=[];rows=[];offset=0
    for repeat in range(2):
        for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=repeat).split(x,y)):
            if fold==2:continue
            groups.append(dict(repeat=repeat,fold=fold,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            rows.extend(dict(repeat=repeat,fold=fold,row_index=int(i),id=f"{i:04d}",target=int(y.iloc[i])) for i in va);offset+=len(va)
    frame=pd.DataFrame(rows);frame["id"]=frame.id.astype("string")
    test=tmp_path/"test.csv";sample=tmp_path/"sample.csv"
    xt=x.iloc[:8].copy();xt.insert(0,"id",[f"T{i:04d}" for i in range(8)]);xt.to_csv(test,index=False)
    pd.DataFrame({"id":xt.id.iloc[::-1].to_numpy(),"target":0.}).to_csv(sample,index=False)
    cfg=dict(test_path=str(test),sample_submission_path=str(sample),expected_test_rows=8)
    return dict(x=x,y=y,features=list(x.columns),ids=pd.Series([f"{i:04d}" for i in range(48)]),frame=frame,groups=groups,
        control=.1+.8*frame.target.to_numpy(),reference=.2+.5*frame.target.to_numpy(),config=dict(id_column="id",target_column="target"),
        hashes={},followup_config=cfg,source_study="synthetic",output=tmp_path/"result",action=action)


def test_plain_pca_does_not_fit_validation():
    x=np.random.RandomState(3).normal(size=(30,10));p=PlainPCAProjector(4).fit(x)
    before=p.scaler_.mean_.copy();assert p.transform(x+100).shape==(30,4)
    np.testing.assert_array_equal(before,p.scaler_.mean_)


def test_fixed_design_and_no_search_driven_export():
    assert f.DESIGN["capacity_k"]==[8,12,16]
    assert f.DESIGN["blend_gmm_weights"]==[.25,.5,.75]
    assert f.CONTROL=="A_k8_tied"


@pytest.mark.parametrize("action,expected_fits",[("capacity",24),("ablation",12),("blend",0)])
def test_real_research_actions(tmp_path,action,expected_fits):
    c=context(tmp_path,action);decision=f.run_research(c)
    m=f.base.read(c["output"]/"run_manifest.json")
    assert m["status"]=="completed" and m["completed_candidate_fits"]==expected_fits
    assert m["competition_test_opened"] is False and decision["automatic_promotion"] is False
    for name,h in m["artifact_sha256"].items():assert f.base.digest(c["output"]/name)==h
    with np.load(c["output"]/"predictions.npz") as z:
        np.testing.assert_array_equal(z[f.CONTROL],c["control"])
        if action=="blend":
            for w in f.DESIGN["blend_gmm_weights"]:np.testing.assert_allclose(z[f"blend_gmm{w:g}"],w*c["control"]+(1-w)*c["reference"])
    if action=="blend":assert len(pd.read_csv(c["output"]/"complementarity.csv"))==4
    with pytest.raises(FileExistsError):f.run_research(c)


def test_export_three_real_fits_reload_and_template_alignment(tmp_path,monkeypatch):
    c=context(tmp_path,"export");original=f.model;fits=[]
    cls=type(original(8,42));fit=cls.fit
    def record(self,x,y,x_unlabelled=None):
        assert x_unlabelled is None and len(x)==48
        value=fit(self,x,y,x_unlabelled);fits.append(self.random_state);return value
    monkeypatch.setattr(cls,"fit",record)
    read=ex.read_csv
    def guarded(*args,**kwargs):
        assert fits==[42,314,2026]
        return read(*args,**kwargs)
    monkeypatch.setattr(ex,"read_csv",guarded)
    result=ex.export(c)
    assert result["fits_performed"]==3 and result["rows"]==8
    m=f.base.read(c["output"]/"run_manifest.json")
    assert m["saved_and_reloaded_models_verified"] is True and m["competition_test_used_for_fitting"] is False
    import joblib
    bundle=joblib.load(c["output"]/"model_bundle.joblib")
    test=pd.read_csv(c["followup_config"]["test_path"],dtype={"id":"string"})
    submission=pd.read_csv(c["output"]/"submission.csv",dtype={"id":"string"})
    assert submission.id.tolist()==list(reversed(test.id.tolist()))
    expected=ex.bundle_predict(bundle,test[c["features"]])[::-1]
    np.testing.assert_allclose(submission.target,expected,rtol=0,atol=1e-12)
    for name,h in m["artifact_sha256"].items():assert f.base.digest(c["output"]/name)==h


def test_nonconverged_export_does_not_open_test(tmp_path,monkeypatch):
    c=context(tmp_path,"export");original=f.model;cls=type(original(8,42));fit=cls.fit
    def nonconverged(self,*a,**kw):
        value=fit(self,*a,**kw);self.converged_=False;return value
    monkeypatch.setattr(cls,"fit",nonconverged)
    monkeypatch.setattr(ex,"read_csv",lambda *a,**kw:pytest.fail("Test was opened"))
    with pytest.raises(RuntimeError,match="did not converge"):ex.export(c)
    assert not (c["output"]/"submission.csv").exists()
    assert f.base.read(c["output"]/"run_manifest.json")["competition_test_opened"] is False


def test_template_and_schema_guards(tmp_path):
    c=context(tmp_path,"export");test=ex.read_csv(c["followup_config"]["test_path"],"id");template=ex.read_csv(c["followup_config"]["sample_submission_path"],"id");p=np.linspace(.1,.9,8)
    bad=template.copy();bad.loc[0,"id"]="missing"
    with pytest.raises(ValueError,match="ID sets differ"):ex.align_submission(test,bad,c["features"],p,8)
    with pytest.raises(ValueError,match="row count"):ex.align_submission(test,template,c["features"],p,9)
    with pytest.raises(ValueError,match="schema"):ex.align_submission(test.assign(target=0),template,c["features"],p,8)


def source_fixture(tmp_path,monkeypatch):
    c=context(tmp_path,"capacity");source=tmp_path/"gmm_source";source.mkdir()
    oldcfg=tmp_path/"old_config.json";f.base.write(oldcfg,c["config"])
    f.base.write(source/"settings.json",dict(config=c["config"],modes=["A","B"],versions={"numpy":f.version("numpy")},input_and_code_sha256={}))
    f.base.write(source/"decision.json",dict(selected_candidate=f.CONTROL,selected_mean_auc=1.,screening_passed=True,eligible_candidates={f.CONTROL:True}))
    f.base.write(source/"candidate_specs.json",{f.CONTROL:dict(mode="A",n_components=8,covariance_type="tied")})
    f.base.write(source/"split_indices.json",c["groups"]);c["frame"].to_csv(source/"cohort.csv",index=False)
    np.savez_compressed(source/"predictions.npz",**{f.CONTROL:c["control"],"reference_trial25":c["reference"]})
    def rehash():
        hashes={p.name:f.base.digest(p) for p in source.iterdir() if p.is_file() and p.name!="run_manifest.json"}
        f.base.write(source/"run_manifest.json",dict(status="completed",planned_candidate_fits=1800,completed_candidate_fits=1800,artifact_sha256=hashes))
    rehash()
    cfg=dict(source_config=str(oldcfg),source_study=str(source),output_root=str(tmp_path/"followup"),submission_output=str(tmp_path/"export"),
        test_path=c["followup_config"]["test_path"],sample_submission_path=c["followup_config"]["sample_submission_path"],expected_test_rows=8,design=f.DESIGN.copy())
    path=tmp_path/"cfg.json";f.base.write(path,cfg)
    def old_prepare(path,output):return dict(c,output=Path(output))
    monkeypatch.setattr(f.base,"prepare",old_prepare);monkeypatch.setattr(f,"EXPECTED_AUC",1.)
    return c,source,path,rehash


def test_preflight_no_write_no_test_reader(tmp_path,monkeypatch):
    c,source,path,rehash=source_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(ex,"read_csv",lambda *a,**kw:pytest.fail("dry preflight opened test"))
    prepared=f.prepare(str(path),"export")
    assert not prepared["output"].exists()
    assert prepared["control"].shape==c["control"].shape


def test_source_artifact_corruption_rejected(tmp_path,monkeypatch):
    c,source,path,rehash=source_fixture(tmp_path,monkeypatch)
    (source/"cohort.csv").write_text("tamper")
    with pytest.raises(ValueError,match="changed generative artifact"):f.prepare(str(path),"capacity")


def test_source_winner_change_rejected(tmp_path,monkeypatch):
    c,source,path,rehash=source_fixture(tmp_path,monkeypatch)
    d=f.base.read(source/"decision.json");d["selected_candidate"]="A_k12_tied";f.base.write(source/"decision.json",d);rehash()
    with pytest.raises(ValueError,match="A_k8_tied winner"):f.prepare(str(path),"export")


def test_reference_identity_change_rejected(tmp_path,monkeypatch):
    c,source,path,rehash=source_fixture(tmp_path,monkeypatch)
    frame=pd.read_csv(source/"cohort.csv",dtype={"id":"string"});frame.loc[0,"id"]="different";frame.to_csv(source/"cohort.csv",index=False);rehash()
    with pytest.raises(ValueError,match="cohort/splits differ"):f.prepare(str(path),"capacity")


def test_environment_change_rejected(tmp_path,monkeypatch):
    c,source,path,rehash=source_fixture(tmp_path,monkeypatch)
    d=f.base.read(source/"settings.json");d["versions"]["numpy"]="not-installed";f.base.write(source/"settings.json",d);rehash()
    with pytest.raises(ValueError,match="environment version changed"):f.prepare(str(path),"capacity")


def test_blend_research_never_reads_competition_files(tmp_path,monkeypatch):
    c=context(tmp_path,"blend")
    Path(c["followup_config"]["test_path"]).unlink();Path(c["followup_config"]["sample_submission_path"]).unlink()
    assert f.run_research(c)["action"]=="blend"

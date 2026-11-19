import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.submission import generative_trial21_snapshot_v1 as s
from src.submission import generative_trial21_export_v1 as e


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):yield


def write(p,data):p.parent.mkdir(parents=True,exist_ok=True);s.b.write(p,data)
def manifest(root,name,fields):
    hashes={str(p.relative_to(root)):s.b.digest(p) for p in root.rglob("*") if p.is_file() and p.name!=name}
    write(root/name,dict(fields,artifact_sha256=hashes))


def export_context(tmp_path):
    x,y=make_classification(n_samples=64,n_features=24,n_informative=8,n_redundant=0,random_state=4)
    x=pd.DataFrame(x,columns=[f"f{i}" for i in range(24)]);y=pd.Series(y)
    test=tmp_path/"test.csv";template=tmp_path/"sample.csv";trial=tmp_path/"trial25";trial.mkdir()
    xt=x.iloc[:8].copy();xt.insert(0,"id",[f"{i:05d}" for i in range(8)]);xt.to_csv(test,index=False)
    pd.DataFrame({"id":xt.id.iloc[::-1].to_numpy(),"target":0.}).to_csv(template,index=False)
    l1=np.linspace(.1,.7,8);xgb=l1+.03;lgb=l1+.06;en=np.linspace(.9,.2,8);base=(l1+xgb+lgb)/3
    old=(1-s.TRIAL25_PARAMETERS["alpha"])*base+s.TRIAL25_PARAMETERS["alpha"]*en
    pd.DataFrame(dict(id=xt.id,l1=l1,xgboost=xgb,lightgbm=lgb,elasticnet=en,original_three_model=base,ensemble=old)).to_csv(trial/"component_predictions.csv",index=False,float_format="%.17g")
    pd.DataFrame({"id":xt.id,"target":old}).iloc[[2,0,7,1,6,3,5,4]].to_csv(trial/"submission.csv",index=False,float_format="%.17g")
    return dict(output=tmp_path/"export",x=x,y=y,hashes={},selected_parameters=s.PARAMETERS,source_best={"parameters":s.PARAMETERS},source_decision={"selected_trial":21},source_summary=pd.DataFrame([dict(candidate="synthetic",mean_auc=.9)]),
        export_config=dict(test_path=str(test),sample_submission_path=str(template),expected_rows=8),trial25_root=trial,
        trial25_manifest=dict(prediction_input_sha256={str(test):s.b.digest(test),str(template):s.b.digest(template)}))


def test_three_native_fits_two_outputs_persistence_and_alignment(tmp_path,monkeypatch):
    c=export_context(tmp_path);build=s.search.model;klass=type(build(s.PARAMETERS,42));fit=klass.fit;calls=[]
    def recorded(self,x,y,*args,**kwargs):
        assert len(x)==64 and not args and not kwargs
        value=fit(self,x,y);calls.append(self.random_state);return value
    monkeypatch.setattr(klass,"fit",recorded)
    get=e.prediction_inputs
    def guarded(c):assert calls==[42,314,2026];return get(c)
    monkeypatch.setattr(e,"prediction_inputs",guarded)
    result=e.execute(c);assert result["fits_performed"]==3
    m=s.b.read(c["output"]/"prediction_manifest.json")
    assert m["status"]=="ready_for_manual_submission" and m["saved_and_reloaded_models_verified"] is True
    assert m["competition_test_used_for_fitting"] is False
    a=s.csv_frame(c["output"]/"standalone/submission.csv");h=s.csv_frame(c["output"]/"hybrid/submission.csv")
    old=s.csv_frame(c["trial25_root"]/"submission.csv").set_index("id").loc[a.id,"target"].to_numpy()
    np.testing.assert_allclose(h.target,.5*a.target+.5*old,rtol=0,atol=1e-12)
    assert a.id.tolist()==[f"{i:05d}" for i in reversed(range(8))]
    for name,hash in m["artifact_sha256"].items():assert s.b.digest(c["output"]/name)==hash
    for path in ["standalone","hybrid"]:assert s.b.read(c["output"]/path/"candidate_manifest.json")["rows"]==8
    with pytest.raises(FileExistsError):e.execute(c)


def test_nonconvergence_before_test_access(tmp_path,monkeypatch):
    c=export_context(tmp_path);cls=type(s.search.model(s.PARAMETERS,42));fit=cls.fit
    def bad(self,*a,**kw):value=fit(self,*a,**kw);self.converged_=False;return value
    monkeypatch.setattr(cls,"fit",bad);monkeypatch.setattr(e,"prediction_inputs",lambda *a:pytest.fail("Test opened"))
    with pytest.raises(RuntimeError,match="did not converge"):e.execute(c)
    assert s.b.read(c["output"]/"prediction_manifest.json")["competition_test_opened"] is False
    assert not list(c["output"].glob("*/submission.csv"))


def test_different_test_inputs_fail_without_submission(tmp_path):
    c=export_context(tmp_path);p=Path(c["export_config"]["test_path"]);df=s.csv_frame(p);df["f0"]+=100;df.to_csv(p,index=False)
    with pytest.raises(ValueError,match="different test/template"):e.execute(c)
    assert not list(c["output"].glob("*/submission.csv"))


def test_two_candidate_publication_rolls_back_on_second_rename(tmp_path,monkeypatch):
    c=export_context(tmp_path);replace=Path.replace
    def fail_second(self,target):
        if self.name=="submission.pending.csv" and self.parent.name=="hybrid":raise OSError("Injected second publication failure")
        return replace(self,target)
    monkeypatch.setattr(Path,"replace",fail_second)
    with pytest.raises(OSError,match="second publication"):e.execute(c)
    assert not list(c["output"].glob("*/submission.csv"))
    assert s.b.read(c["output"]/"prediction_manifest.json")["status"]=="failed_or_interrupted"


def source_context(tmp_path,monkeypatch):
    root=tmp_path/"study";root.mkdir();trialroot=tmp_path/"trial25";trialroot.mkdir();original=tmp_path/"original_trial25"
    y=np.tile([0,1],125);rows=[];groups=[];offset=0
    for repeat in range(10):
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=repeat).split(np.zeros((250,1)),y)):
            groups.append(dict(repeat=repeat,fold=fold,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            rows.extend(dict(repeat=repeat,fold=fold,row_index=int(i),id=f"L{i:04d}",target=int(y[i])) for i in va);offset+=len(va)
    frame=pd.DataFrame(rows);frame["id"]=frame.id.astype("string");rng=np.random.RandomState(25);signal=2*frame.target.to_numpy()-1
    old=expit(signal+rng.normal(size=2500));ref=expit(.95*signal+rng.normal(size=2500));new=expit(3*signal+rng.normal(size=2500))
    train=tmp_path/"train.csv";x=pd.DataFrame(np.zeros((250,150)),columns=[f"f{i}" for i in range(150)]);x.assign(id=[f"L{i:04d}" for i in range(250)],target=y).to_csv(train,index=False)
    od=dict(selected_trial=25,parameters=s.TRIAL25_PARAMETERS,screening_both_controls=True);ob=dict(number=25,state="COMPLETE",parameters=s.TRIAL25_PARAMETERS)
    write(original/"decision.json",od);write(original/"best_trial.json",ob)
    hashes={str(train):s.b.digest(train),str(original/"decision.json"):s.b.digest(original/"decision.json"),str(original/"best_trial.json"):s.b.digest(original/"best_trial.json")}
    c=dict(x=x,y=pd.Series(y),frame=frame,groups=groups,control=old,reference=ref,hybrid_control=.5*old+.5*ref,hashes=hashes,
        optuna_config={"synthetic":True},config=dict(id_column="id",target_column="target",train_path=str(train)))
    results={};summaries={}
    for number,raw in [(0,old),(21,new)]:
        directory=root/"trials"/f"trial_{number:04d}";directory.mkdir(parents=True)
        params=s.search.CONTROL if number==0 else s.PARAMETERS;hybrid=.5*raw+.5*ref
        preds=dict(candidate_gmm=raw,candidate_hybrid=hybrid,original_gmm=old,original_hybrid=c["hybrid_control"],reference_trial25=ref)
        _,rr,ss=s.b.evaluate(c,preds);means=ss.set_index("candidate").mean_auc;summaries[number]=ss
        pairs=[("candidate_hybrid",name) for name in ["original_hybrid","original_gmm","reference_trial25"]]+[("candidate_gmm","original_gmm")]
        result=dict(number=number,state="COMPLETE",parameters=params,objective_auc=float(means["candidate_hybrid"]),standalone_auc=float(means["candidate_gmm"]),comparisons=[s.b.compare(rr,a,z) for a,z in pairs],control_reproduced=True if number==0 else None)
        results[number]=result;write(directory/"parameters.json",params);write(directory/"result.json",result);ss.to_csv(directory/"model_summary.csv",index=False);np.savez_compressed(directory/"predictions.npz",gmm=raw,hybrid=hybrid)
        fits=[dict(repeat=g["repeat"],fold=g["fold"],seed=seed,converged=True,error=None) for g in groups for seed in s.b.DESIGN["initialization_seeds"]]
        (directory/"fit_diagnostics.jsonl").write_text("".join(json.dumps(v)+"\n" for v in fits))
        geometry=[dict(repeat=g["repeat"],fold=g["fold"],requested_features=params["selected_features"],requested_dimensions=params["latent_dimensions"],effective_features=20,effective_dimensions=params["latent_dimensions"],selected_indices=list(range(20))) for g in groups]
        (directory/"geometry_diagnostics.jsonl").write_text("".join(json.dumps(v)+"\n" for v in geometry))
    best=results[21];monkeypatch.setattr(s,"GMM_AUC",best["standalone_auc"]);monkeypatch.setattr(s,"HYBRID_AUC",best["objective_auc"]);monkeypatch.setattr(s,"PINS",{})
    trials=[dict(number=i,state="COMPLETE",value=best["objective_auc"] if i in (21,24) else results[0]["objective_auc"],parameters=s.PARAMETERS if i in (21,24) else s.search.CONTROL,user_attrs={"control_reproduced":True} if i==0 else {}) for i in range(30)]
    write(root/"trials.json",trials);pd.DataFrame([dict(number=t["number"],state=t["state"],objective_auc=t["value"],**t["parameters"]) for t in trials]).to_csv(root/"trials.csv",index=False)
    write(root/"best_trial.json",best);write(root/"decision.json",dict(selected_trial=21,selected_parameters=s.PARAMETERS,selected_mean_auc=best["objective_auc"],selected_is_control=False,screening_passed=True,automatic_promotion=False,comparisons=best["comparisons"]))
    write(root/"settings.json",dict(config=c["optuna_config"],fixed_source_design=s.b.DESIGN,hybrid_weights=[.5,.5],input_and_code_sha256=hashes,versions={},optuna_version="fixture"))
    frame.to_csv(root/"cohort.csv",index=False);write(root/"split_indices.json",groups)
    fields=dict(status="completed",planned_trials=30,attempted_trials=30,completed_trials=30,rejected_trials=0,failed_trials=0,planned_candidate_fits=4500,attempted_candidate_fits=4500,completed_candidate_fits=4500,converged_candidate_fits=4500,attempted_projector_fits=1500,completed_projector_fits=1500,competition_test_opened=False,kaggle_upload_performed=False)
    def rehash():manifest(root,"run_manifest.json",fields)
    rehash();write(trialroot/"source_decision.json",od);write(trialroot/"source_best_trial.json",ob)
    write(trialroot/"ensemble.json",dict(elasticnet_parameters=s.TRIAL25_PARAMETERS,feature_order=list(x.columns)))
    pd.DataFrame({"id":[str(i) for i in range(8)],"target":.5}).to_csv(trialroot/"submission.csv",index=False)
    (trialroot/"component_predictions.csv").write_text("id,l1\n0,0.5\n")
    test=tmp_path/"test.csv";test.write_text("id,f0\n0,0\n");template=tmp_path/"sample.csv";template.write_text("id,target\n0,0\n")
    tfields=dict(status="ready_for_manual_submission",candidate="elasticnet_optuna_trial_25",selected_trial=25,parameters=s.TRIAL25_PARAMETERS,fits_performed=5,saved_and_reloaded_models_verified=True,rows=8,kaggle_upload_performed=False,source_study=str(original),source_input_and_code_sha256=hashes,prediction_input_sha256={str(test):s.b.digest(test),str(template):s.b.digest(template)},submission_sha256=s.b.digest(trialroot/"submission.csv"))
    manifest(trialroot,"prediction_manifest.json",tfields)
    config=tmp_path/"cfg.json";write(config,dict(study=str(root),study_config="synthetic",trial25_export=str(trialroot),test_path=str(test),sample_submission_path=str(template),expected_rows=8,output_dir=str(tmp_path/"new")))
    monkeypatch.setattr(s.search.snapshot,"prepare",lambda config,output:dict(c,output=Path(output)))
    v=s.version;monkeypatch.setattr(s,"version",lambda p:"fixture" if p=="optuna" else v(p))
    return config,root,rehash,test


def test_complete_source_verification_and_no_test_reads(tmp_path,monkeypatch):
    config,root,rehash,test=source_context(tmp_path,monkeypatch)
    digest=s.b.digest
    def guard(p):
        if Path(p).name in ("test.csv","sample.csv"):pytest.fail("Prediction input opened during dry source verification")
        return digest(p)
    monkeypatch.setattr(s.b,"digest",guard)
    c=s.prepare(str(config));assert c["source_best"]["number"]==21 and not c["output"].exists()


def test_pinned_fingerprint_mismatch_is_fatal(tmp_path,monkeypatch):
    config,root,rehash,test=source_context(tmp_path,monkeypatch);monkeypatch.setattr(s,"PINS",{"best_trial.json":"0"*64})
    with pytest.raises(ValueError,match="fingerprint"):s.prepare(str(config))


def test_source_tamper_is_fatal(tmp_path,monkeypatch):
    config,root,rehash,test=source_context(tmp_path,monkeypatch);(root/"trials.csv").write_text("corrupt")
    with pytest.raises(ValueError,match="Changed source"):s.prepare(str(config))


def test_rehashed_wrong_winner_is_fatal(tmp_path,monkeypatch):
    config,root,rehash,test=source_context(tmp_path,monkeypatch);d=s.b.read(root/"decision.json");d["selected_trial"]=24;write(root/"decision.json",d);rehash()
    with pytest.raises(ValueError,match="decision differs"):s.prepare(str(config))


def test_rehashed_row_prediction_mismatch_is_fatal(tmp_path,monkeypatch):
    config,root,rehash,test=source_context(tmp_path,monkeypatch);p=root/"trials/trial_0021/predictions.npz"
    with np.load(p) as z:g=z["gmm"].copy();h=z["hybrid"].copy()
    g[0]=1-g[0];np.savez_compressed(p,gmm=g,hybrid=h);rehash()
    with pytest.raises(ValueError,match="formula differs"):s.prepare(str(config))


def test_no_output_overlap(tmp_path,monkeypatch):
    config,root,rehash,test=source_context(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="overlaps source"):s.prepare(str(config),str(root/"new"))


def test_cli_dry_run_zero_fit_no_output(tmp_path,monkeypatch,capsys):
    import sys
    monkeypatch.setattr(s,"prepare",lambda *a:dict(output=tmp_path/"none"))
    monkeypatch.setattr(sys,"argv",["export-trial21","--dry-run"]);e.main();r=json.loads(capsys.readouterr().out)
    assert r["fits_performed"]==0 and r["competition_test_opened"] is False and not Path(r["output"]).exists()


def test_incomplete_fit_accounting_is_fatal(tmp_path,monkeypatch):
    config,root,rehash,test=source_context(tmp_path,monkeypatch)
    m=s.b.read(root/"run_manifest.json");m["converged_candidate_fits"]=4499;write(root/"run_manifest.json",m)
    with pytest.raises(ValueError,match="4500-fit"):s.prepare(str(config))


def test_rehashed_bad_geometry_dimension_is_fatal(tmp_path,monkeypatch):
    config,root,rehash,test=source_context(tmp_path,monkeypatch);p=root/"trials/trial_0021/geometry_diagnostics.jsonl"
    rows=[json.loads(v) for v in p.read_text().splitlines()];rows[0]["effective_dimensions"]=8;p.write_text("".join(json.dumps(v)+"\n" for v in rows));rehash()
    with pytest.raises(ValueError,match="effective dimension"):s.prepare(str(config))

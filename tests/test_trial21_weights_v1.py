import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from src.experiments import trial21_weights_v1 as w


def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);w.b.write(p,x)
def manifest(root,name,fields):
    write(root/name,dict(fields,artifact_sha256={str(p.relative_to(root)):w.b.digest(p) for p in root.rglob("*") if p.is_file() and p.name!=name}))


@pytest.fixture
def context(tmp_path,monkeypatch):
    y=np.tile([0,1],125);rows=[];groups=[];offset=0
    for r in range(10):
        for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(np.zeros((250,1)),y)):
            groups.append(dict(repeat=r,fold=f,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            rows.extend(dict(repeat=r,fold=f,row_index=int(i),id=f"L{i:04d}",target=int(y[i])) for i in va);offset+=len(va)
    frame=pd.DataFrame(rows);frame["id"]=frame.id.astype("string");rng=np.random.RandomState(25);signal=2*frame.target.to_numpy()-1
    new=expit(2.6*signal+rng.normal(size=2500));old=expit(signal+rng.normal(size=2500));ref=expit(.95*signal+rng.normal(size=2500))
    study=tmp_path/"study";p=study/"trials/trial_0021/predictions.npz";p.parent.mkdir(parents=True);np.savez_compressed(p,gmm=new,hybrid=.5*new+.5*ref)
    train=tmp_path/"train.csv";train.write_text("immutable training input")
    sourcecfg=tmp_path/"sourcecfg.json";write(sourcecfg,{"fixture":True})
    cfg=dict(trial21_export_config=str(sourcecfg),trial21_export=str(tmp_path/"trial21_export"),old_hybrid_export=str(tmp_path/"old_hybrid_export"),comparison_output=str(tmp_path/"comparison"),export_output=str(tmp_path/"selected_export"),design=w.DESIGN)
    config=tmp_path/"cfg.json";write(config,cfg)
    c=dict(frame=frame,groups=groups,control=old,reference=ref,hybrid_control=.5*old+.5*ref,new_gmm=new,new_hybrid=.5*new+.5*ref,
        study_root=study,source_study=str(tmp_path/"original_geometry"),source_best={"number":21},source_decision={"selected_trial":21},
        hashes={str(p):w.b.digest(p),str(train):w.b.digest(train)},trial25_root=tmp_path/"trial25",export_config=dict(test_path=str(tmp_path/"test.csv"),sample_submission_path=str(tmp_path/"sample.csv"),expected_rows=8),
        weights_config=cfg,output=tmp_path/"comparison",config_path=config)
    monkeypatch.setattr(w.source,"prepare",lambda config,output:dict(c,output=Path(output),hashes=dict(c["hashes"])))
    return c


def test_fixed_formulas_effective_weights_and_alias(context):
    preds,_,_,_,_,_=w.calculate(context);assert len(w.specs())==9
    for a in w.DESIGN["alphas"]:
        np.testing.assert_allclose(preds[f"A_a{a:g}"],a*context["new_gmm"]+(1-a)*context["hybrid_control"],rtol=0,atol=0)
        np.testing.assert_allclose(preds[f"B_a{a:g}"],a*context["new_gmm"]+(1-a)*context["reference"],rtol=0,atol=0)
    np.testing.assert_allclose(preds["B_a0.5"],context["new_hybrid"],atol=1e-12,rtol=0)
    for spec in w.specs().values():assert sum(spec["effective_weights"].values())==pytest.approx(1)
    assert "H_new" not in preds


def selection(means,primary_wins=9):
    comparisons=[]
    for name in w.specs():
        for ref in ["new_gmm","old_hybrid","B_a0.5"]:
            if name!=ref:comparisons.append(dict(candidate=name,reference=ref,mean_difference=float(means[name]-(means.get(ref,.8))),difference_sd=.01,wins=primary_wins if ref=="new_gmm" else 10,losses=10-primary_wins if ref=="new_gmm" else 0,ties=0))
    return w.select(means,comparisons)


def test_standalone_tie_priority():
    result=selection({n:.9 for n in w.specs()})
    assert result["selected_candidate"]=="new_gmm" and result["screening_passed"] is None


def test_blend_pass():
    means={n:.9 for n in w.specs()};means["A_a0.9"]=.91
    result=selection(means)
    assert result["retained_candidate"]=="A_a0.9" and result["screening_passed"] is True


def test_failure_retains_standalone_not_second_blend():
    means={n:.9 for n in w.specs()};means["A_a0.9"]=.92;means["B_a0.95"]=.91
    result=selection(means,primary_wins=4)
    assert result["selected_candidate"]=="A_a0.9" and result["retained_candidate"]=="new_gmm"


def test_blend_ties_use_A_then_alpha_order():
    means={n:.9 for n in w.specs()};means.update({"A_a0.5":.92,"A_a0.95":.92,"B_a0.5":.92})
    assert selection(means)["selected_candidate"]=="A_a0.5"


def test_real_compare_and_frozen_replay(context):
    c=w.prepare(str(context["config_path"]),"compare");result=w.compare(c)
    m=w.b.read(c["output"]/"run_manifest.json")
    assert m["fits_performed"]==0 and m["competition_test_opened"] is False and m["status"]=="completed"
    for name,h in m["artifact_sha256"].items():assert w.b.digest(c["output"]/name)==h
    export=w.prepare(str(context["config_path"]),"export")
    assert w.frozen_comparison(export)==result
    with pytest.raises(FileExistsError):w.prepare(str(context["config_path"]),"compare")


def test_rehashed_wrong_comparison_decision_fails(context):
    c=w.prepare(str(context["config_path"]),"compare");w.compare(c)
    d=w.b.read(c["output"]/"decision.json");d["retained_candidate"]="A_a0.5" if d["retained_candidate"]!="A_a0.5" else "new_gmm";write(c["output"]/"decision.json",d)
    manifest(c["output"],"run_manifest.json",dict(status="completed",action="compare",fits_performed=0,competition_test_opened=False))
    with pytest.raises(ValueError,match="decision failed"):w.frozen_comparison(w.prepare(str(context["config_path"]),"export"))


def test_compare_does_not_read_competition_files(context):
    assert not Path(context["export_config"]["test_path"]).exists()
    c=w.prepare(str(context["config_path"]),"compare");assert w.compare(c)["automatic_promotion"] is False


def export_sources(c):
    cfg=c["weights_config"];newroot=Path(cfg["trial21_export"]);oldroot=Path(cfg["old_hybrid_export"]);tr=c["trial25_root"];gmroot=newroot.parent/"old_gmm_export"
    for p in [newroot,oldroot,tr,gmroot]:p.mkdir()
    ids=[f"{i:05d}" for i in range(8)];test=Path(c["export_config"]["test_path"]);sample=Path(c["export_config"]["sample_submission_path"])
    pd.DataFrame({"id":ids,"feature":np.arange(8)}).to_csv(test,index=False);order=[3,0,7,1,6,2,5,4]
    pd.DataFrame({"id":[ids[i] for i in order],"target":0.}).to_csv(sample,index=False)
    inputs={str(p):w.b.digest(p) for p in [test,sample]};l1=np.linspace(.1,.7,8);xgb=l1+.03;lgb=l1+.06;en=np.linspace(.9,.2,8);base=(l1+xgb+lgb)/3;t=(1-w.source.TRIAL25_PARAMETERS["alpha"])*base+w.source.TRIAL25_PARAMETERS["alpha"]*en
    g=np.linspace(.8,.1,8);new=np.array([.01,.04,.12,.4,.65,.9,.97,.99]);oldhyb=.5*g+.5*t;newhyb=.5*new+.5*t
    def submission(root,relative,p):
        file=root/relative;file.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({"id":ids,"target":p}).iloc[::-1].to_csv(file,index=False,float_format="%.17g")
    submission(newroot,"standalone/submission.csv",new);submission(newroot,"hybrid/submission.csv",newhyb)
    write(newroot/"standalone/candidate_manifest.json",{"candidate":"trial21"});write(newroot/"settings.json",dict(parameters=w.source.PARAMETERS,seeds=[42,314,2026],covariance="tied"));write(newroot/"source_best_trial.json",c["source_best"]);write(newroot/"source_decision.json",c["source_decision"])
    pd.DataFrame(dict(id=ids,trial21_gmm=new,trial25=t,trial21_hybrid=newhyb)).to_csv(newroot/"aligned_predictions.csv",index=False,float_format="%.17g");(newroot/"model_bundle.joblib").write_bytes(b"must not deserialize")
    manifest(newroot,"prediction_manifest.json",dict(status="ready_for_manual_submission",selected_trial=21,parameters=w.source.PARAMETERS,completed_candidate_fits=3,saved_and_reloaded_models_verified=True,rows=8,competition_test_used_for_fitting=False,kaggle_upload_performed=False,source_input_and_code_sha256={},prediction_input_sha256=inputs))
    submission(tr,"submission.csv",t);write(tr/"source_best_trial.json",{"number":25})
    pd.DataFrame(dict(id=ids,l1=l1,xgboost=xgb,lightgbm=lgb,elasticnet=en,original_three_model=base,ensemble=t)).to_csv(tr/"component_predictions.csv",index=False,float_format="%.17g")
    manifest(tr,"prediction_manifest.json",dict(status="ready_for_manual_submission",selected_trial=25,parameters=w.source.TRIAL25_PARAMETERS,saved_and_reloaded_models_verified=True,source_input_and_code_sha256={},prediction_input_sha256=inputs))
    submission(gmroot,"submission.csv",g);(gmroot/"model_bundle.joblib").write_bytes(b"also never load")
    write(gmroot/"settings.json",dict(source_study=c["source_study"],source_design=w.b.DESIGN,input_and_code_sha256={}))
    manifest(gmroot,"run_manifest.json",dict(status="ready_for_manual_submission",action="export",completed_candidate_fits=3,saved_and_reloaded_models_verified=True,export_input_sha256=inputs))
    submission(oldroot,"submission.csv",oldhyb)
    pd.DataFrame(dict(id=ids,gmm_k8=g,trial25=t,blend=oldhyb)).to_csv(oldroot/"aligned_source_predictions.csv",index=False,float_format="%.17g")
    _,_,ss=w.b.evaluate(c,{"old":c["hybrid_control"]})
    write(oldroot/"source_cv_decision.json",dict(selected_candidate="blend_gmm0.5",selected_mean_auc=float(ss.mean_auc.iloc[0])))
    write(oldroot/"lineage.json",dict(sources={"trial25_export":dict(submission_sha256=w.b.digest(tr/"submission.csv")),"gmm_export":dict(directory=str(gmroot),manifest_sha256=w.b.digest(gmroot/"run_manifest.json"),submission_sha256=w.b.digest(gmroot/"submission.csv"))}))
    manifest(oldroot,"prediction_manifest.json",dict(status="ready_for_manual_submission",candidate="blend_gmm0.5",weights=dict(gmm_weight=.5,trial25_weight=.5),fits_performed=0,source_input_and_code_sha256={},official_input_sha256=inputs))
    return dict(new=new,g=g,t=t,oldhyb=oldhyb,order=order)


@pytest.mark.parametrize("candidate",["new_gmm","A_a0.5","A_a0.95","B_a0.75"])
def test_zero_fit_export_each_family_and_alignment(context,candidate):
    c=w.prepare(str(context["config_path"]),"export");p=export_sources(c)
    d=dict(retained_candidate=candidate,retained_spec=w.specs()[candidate]);result=w.export(c,d)
    assert result["fits_performed"]==0
    frame=w.source.csv_frame(c["output"]/"submission.csv");spec=w.specs()[candidate];a=spec["alpha"]
    expected=p["new"] if spec["family"]=="control" else a*p["new"]+(1-a)*(p["oldhyb"] if spec["family"]=="A" else p["t"])
    np.testing.assert_allclose(frame.target,expected[p["order"]],atol=1e-12,rtol=0)
    assert frame.id.tolist()==[f"{i:05d}" for i in p["order"]]
    m=w.b.read(c["output"]/"prediction_manifest.json");assert m["status"]=="ready_for_manual_submission" and m["competition_test_used_for_fitting"] is False
    for name,h in m["artifact_sha256"].items():assert w.b.digest(c["output"]/name)==h


def test_missing_old_source_fails_instead_of_refitting(context):
    c=w.prepare(str(context["config_path"]),"export");export_sources(c);Path(c["weights_config"]["old_hybrid_export"],"prediction_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):w.export(c,dict(retained_candidate="A_a0.9",retained_spec=w.specs()["A_a0.9"]))
    assert not (c["output"]/"submission.csv").exists()


def test_B_does_not_need_old_hybrid_files(context):
    c=w.prepare(str(context["config_path"]),"export");export_sources(c)
    import shutil
    shutil.rmtree(c["weights_config"]["old_hybrid_export"])
    assert w.export(c,dict(retained_candidate="B_a0.9",retained_spec=w.specs()["B_a0.9"]))["fits_performed"]==0


def test_hash_tamper_export_fails_without_csv(context):
    c=w.prepare(str(context["config_path"]),"export");export_sources(c);(Path(c["weights_config"]["trial21_export"])/"standalone/submission.csv").write_text("tampered")
    with pytest.raises(ValueError,match="Changed source"):w.export(c,dict(retained_candidate="new_gmm",retained_spec=w.specs()["new_gmm"]))
    assert not (c["output"]/"submission.csv").exists()


def test_duplicate_ids_export_fails(context):
    c=w.prepare(str(context["config_path"]),"export");export_sources(c);test=Path(c["export_config"]["test_path"])
    df=pd.read_csv(test,dtype={"id":"string"});df.loc[0,"id"]=df.loc[1,"id"];df.to_csv(test,index=False)
    with pytest.raises(ValueError):w.export(c,dict(retained_candidate="new_gmm",retained_spec=w.specs()["new_gmm"]))
    assert not (c["output"]/"submission.csv").exists()


def test_no_training_or_pickle_dependency():
    text=Path(w.__file__).read_text();assert ".fit(" not in text and "import joblib" not in text and "import pickle" not in text


def test_changed_source_blocks_compare(context):
    c=w.prepare(str(context["config_path"]),"compare");(c["study_root"]/"trials/trial_0021/predictions.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="Changed source"):w.compare(c)
    assert not (c["output"]/"decision.json").exists()


def test_dry_run_no_output_no_test_reads(context,monkeypatch,capsys):
    import sys
    monkeypatch.setattr(sys,"argv",["weights","compare","--config",str(context["config_path"]),"--dry-run"])
    w.main();r=json.loads(capsys.readouterr().out)
    assert r["fits_performed"]==0 and r["competition_test_opened"] is False and not Path(r["output"]).exists()


def test_end_to_end_compare_then_selected_export(context):
    c=w.prepare(str(context["config_path"]),"compare");decision=w.compare(c)
    e=w.prepare(str(context["config_path"]),"export");export_sources(e)
    frozen=w.frozen_comparison(e)
    assert frozen==decision
    result=w.export(e,frozen)
    assert result["candidate"]==decision["retained_candidate"] and result["fits_performed"]==0


def test_export_dry_run_defers_test_input_reads(context,monkeypatch,capsys):
    import sys
    w.compare(w.prepare(str(context["config_path"]),"compare"))
    c=w.prepare(str(context["config_path"]),"export");export_sources(c)
    original=w.b.digest
    def guard(p):
        if Path(p).name in ("test.csv","sample.csv"):pytest.fail("Test opened in dry-run")
        return original(p)
    monkeypatch.setattr(w.b,"digest",guard)
    monkeypatch.setattr(sys,"argv",["weights","export","--config",str(context["config_path"]),"--dry-run"])
    w.main();r=json.loads(capsys.readouterr().out)
    assert r["fits_performed"]==0 and r["competition_test_opened"] is False and not c["output"].exists()


def test_export_dry_run_requires_existing_trial21_predictions(context,monkeypatch):
    import sys
    w.compare(w.prepare(str(context["config_path"]),"compare"))
    monkeypatch.setattr(sys,"argv",["weights","export","--config",str(context["config_path"]),"--dry-run"])
    with pytest.raises(FileNotFoundError,match="Finish existing"):w.main()

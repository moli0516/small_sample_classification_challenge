import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from src.submission import generative_blend_export_v1 as b


def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);b.write(p,x)
def hash_manifest(root,name,fields):
    table={str(p.relative_to(root)):b.digest(p) for p in root.rglob("*") if p.is_file() and p.name!=name}
    write(root/name,dict(fields,artifact_sha256=table))


@pytest.fixture
def sources(tmp_path,monkeypatch):
    train=tmp_path/"train.csv";test=tmp_path/"test.csv";template=tmp_path/"sample.csv"
    pd.DataFrame({"id":[f"L{i:04d}" for i in range(250)],"f0":np.arange(250),"target":np.tile([0,1],125)}).to_csv(train,index=False)
    ids=[f"{i:05d}" for i in range(8)]
    pd.DataFrame({"id":ids,"f0":np.arange(8)}).to_csv(test,index=False)
    order=[3,0,7,1,6,2,5,4]
    pd.DataFrame({"id":[ids[i] for i in order],"target":0.}).to_csv(template,index=False)
    gen=tmp_path/"original_gmm";gd=gen/"decision.json"
    cv=tmp_path/"cv";gmm=tmp_path/"gmm";trial=tmp_path/"trial25"
    for p in [cv,gmm,trial]:p.mkdir()
    y=np.tile([0,1],125);rows=[];groups=[];offset=0
    for r in range(10):
        for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(np.zeros((250,1)),y)):
            groups.append(dict(repeat=r,fold=f,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            rows.extend(dict(repeat=r,fold=f,row_index=int(i),id=f"L{i:04d}",target=int(y[i])) for i in va);offset+=len(va)
    frame=pd.DataFrame(rows);rng=np.random.RandomState(25);signal=2*frame.target.to_numpy()-1
    p_g=expit(signal+rng.normal(0,1,len(frame)));p_t=expit(.95*signal+rng.normal(0,1,len(frame)))
    arrays={"A_k8_tied":p_g,"reference_trial25":p_t}
    for w in [.25,.5,.75]:arrays[f"blend_gmm{w:g}"]=w*p_g+(1-w)*p_t
    scores={n:b.score(frame,groups,p) for n,p in arrays.items()}
    means={n:float(v.mean()) for n,v in scores.items()}
    assert max(["blend_gmm0.25","blend_gmm0.5","blend_gmm0.75"],key=means.get)==b.NAME
    comparisons=[]
    for ref in ["A_k8_tied","reference_trial25"]:
        stat=b.comparison(scores[b.NAME],scores[ref]);assert stat["wins"]>=6 and stat["mean_difference"]>0
        comparisons.append(dict(candidate=b.NAME,reference=ref,**stat))
    write(gd,dict(selected_candidate="A_k8_tied",selected_spec=dict(mode="A",n_components=8,covariance_type="tied"),eligible_candidates={"A_k8_tied":True},screening_passed=True,selected_mean_auc=means["A_k8_tied"]))
    watch={str(train):b.digest(train),str(gd):b.digest(gd)}
    design=dict(initialization_seeds=[42,314,2026],selected_features=20,latent_dimensions=8)
    write(cv/"settings.json",dict(source_study=str(gen),source_design=design,input_and_code_sha256=watch))
    write(cv/"decision.json",dict(action="blend",selected_candidate=b.NAME,selected_spec=b.DESIGN,selected_mean_auc=means[b.NAME],screening_against_both_controls=True,automatic_promotion=False,comparisons=comparisons))
    write(cv/"candidate_specs.json",{b.NAME:b.DESIGN})
    frame.to_csv(cv/"cohort.csv",index=False);write(cv/"split_indices.json",groups);np.savez_compressed(cv/"predictions.npz",**arrays)
    cv_fields=dict(status="completed",action="blend",planned_candidate_fits=0,completed_candidate_fits=0,competition_test_opened=False)
    hash_manifest(cv,"run_manifest.json",cv_fields)
    inputs={str(test):b.digest(test),str(template):b.digest(template)}
    pg=np.linspace(.8,.1,8);gdf=pd.DataFrame({"id":ids,"target":pg}).iloc[::-1]
    gdf.to_csv(gmm/"submission.csv",index=False,float_format="%.17g")
    write(gmm/"settings.json",dict(source_study=str(gen),source_design=design,input_and_code_sha256=watch))
    (gmm/"model_bundle.joblib").write_bytes(b"not a pickle; this exporter must only hash it")
    gfields=dict(status="ready_for_manual_submission",rows=8,action="export",planned_candidate_fits=3,completed_candidate_fits=3,saved_and_reloaded_models_verified=True,competition_test_used_for_fitting=False,kaggle_upload_performed=False,export_input_sha256=inputs)
    hash_manifest(gmm,"run_manifest.json",gfields)
    l1=np.linspace(.1,.7,8);xgb=l1+.03;lgb=l1+.06;en=np.linspace(.9,.2,8);original=(l1+xgb+lgb)/3;pt=(1-b.PARAMETERS["alpha"])*original+b.PARAMETERS["alpha"]*en
    audit=pd.DataFrame(dict(id=ids,l1=l1,xgboost=xgb,lightgbm=lgb,elasticnet=en,original_three_model=original,ensemble=pt))
    audit.to_csv(trial/"component_predictions.csv",index=False,float_format="%.17g")
    pd.DataFrame({"id":ids,"target":pt}).iloc[[2,5,1,7,0,3,6,4]].to_csv(trial/"submission.csv",index=False,float_format="%.17g")
    write(trial/"source_best_trial.json",dict(number=25,state="COMPLETE",parameters=b.PARAMETERS,objective_auc=means["reference_trial25"]))
    write(trial/"source_decision.json",dict(selected_trial=25,parameters=b.PARAMETERS,screening_both_controls=True))
    write(trial/"ensemble.json",dict(candidate="elasticnet_optuna_trial_25",elasticnet_parameters=b.PARAMETERS,alpha=b.PARAMETERS["alpha"]))
    (trial/"models").mkdir();(trial/"models/model.joblib").write_bytes(b"never load this either")
    tfields=dict(status="ready_for_manual_submission",rows=8,candidate="elasticnet_optuna_trial_25",selected_trial=25,parameters=b.PARAMETERS,fits_performed=5,saved_and_reloaded_models_verified=True,kaggle_upload_performed=False,source_input_and_code_sha256=watch,prediction_input_sha256=inputs,submission_sha256=b.digest(trial/"submission.csv"))
    hash_manifest(trial,"prediction_manifest.json",tfields)
    cfg=dict(gmm_export=str(gmm),trial25_export=str(trial),blend_study=str(cv),train_path=str(train),test_path=str(test),sample_submission_path=str(template),expected_rows=8,output_dir=str(tmp_path/"blend_export"),design=b.DESIGN)
    config=tmp_path/"config.json";write(config,cfg);monkeypatch.setattr(b,"EXPECTED_CV_AUC",means[b.NAME])
    return dict(config=config,cfg=cfg,gmm=gmm,trial=trial,cv=cv,gfields=gfields,tfields=tfields,cv_fields=cv_fields,expected=.5*pg+.5*pt,ids=ids,template_order=order)


def test_real_csv_alignment_zero_fit_export_and_hashes(sources):
    s=sources;c=b.prepare(str(s["config"]))
    assert not c["output"].exists()
    assert c["alignment"]["gmm_export"]["reordered_to_template"] is True
    assert c["alignment"]["trial25_export"]["reordered_to_template"] is True
    before={str(p):b.digest(p) for root in [s["gmm"],s["trial"],s["cv"]] for p in root.rglob("*") if p.is_file()}
    result=b.export(c);assert result["fits_performed"]==0
    submission=b.csv_frame(c["output"]/"submission.csv",["id","target"])
    assert submission.id.tolist()==[s["ids"][i] for i in s["template_order"]]
    np.testing.assert_allclose(submission.target,s["expected"][s["template_order"]],atol=1e-12,rtol=0)
    m=b.read(c["output"]/"prediction_manifest.json")
    assert m["status"]=="ready_for_manual_submission" and m["kaggle_upload_performed"] is False
    assert m["competition_test_used_for_fitting"] is False
    for name,h in m["artifact_sha256"].items():assert b.digest(c["output"]/name)==h
    b.check(before)
    with pytest.raises(FileExistsError):b.export(c)


def test_no_fit_pickle_or_model_prediction_code():
    text=Path(b.__file__).read_text()
    assert "import joblib" not in text and "import pickle" not in text and ".fit(" not in text and ".predict_proba(" not in text


def test_manifestless_csv_rejected(sources):
    (sources["gmm"]/"run_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):b.prepare(str(sources["config"]))


def test_tampered_source_csv_rejected(sources):
    with (sources["trial"]/"submission.csv").open("a") as stream:stream.write("wrong,0.2\n")
    with pytest.raises(ValueError,match="Changed/missing source"):b.prepare(str(sources["config"]))


def test_rehashed_trial25_wrong_formula_rejected(sources):
    s=sources;df=b.csv_frame(s["trial"]/"submission.csv");df["target"]+=.001;df.to_csv(s["trial"]/"submission.csv",index=False)
    fields=dict(s["tfields"],submission_sha256=b.digest(s["trial"]/"submission.csv"));hash_manifest(s["trial"],"prediction_manifest.json",fields)
    with pytest.raises(ValueError,match="audit/submission mismatch"):b.prepare(str(s["config"]))


def test_duplicate_source_ids_rejected(sources):
    s=sources;df=b.csv_frame(s["gmm"]/"submission.csv");df.loc[0,"id"]=df.loc[1,"id"];df.to_csv(s["gmm"]/"submission.csv",index=False)
    hash_manifest(s["gmm"],"run_manifest.json",s["gfields"])
    with pytest.raises(ValueError,match="duplicate IDs"):b.prepare(str(s["config"]))


def test_different_official_test_features_rejected(sources):
    s=sources;other=Path(s["cfg"]["test_path"]).with_name("other_test.csv")
    df=pd.read_csv(s["cfg"]["test_path"],dtype={"id":"string"});df["f0"]+=100;df.to_csv(other,index=False)
    cfg=dict(s["cfg"],test_path=str(other));write(s["config"],cfg)
    with pytest.raises(ValueError,match="different test/template"):b.prepare(str(s["config"]))


def test_wrong_candidate_recipe_rejected(sources):
    s=sources;fields=dict(s["gfields"],completed_candidate_fits=2);hash_manifest(s["gmm"],"run_manifest.json",fields)
    with pytest.raises(ValueError,match="three-seed K8"):b.prepare(str(s["config"]))


def test_cv_formula_change_rejected(sources):
    s=sources
    with np.load(s["cv"]/"predictions.npz") as z:arrays={k:z[k].copy() for k in z.files}
    arrays[b.NAME]=np.minimum(arrays[b.NAME]+.0001,.999999);np.savez_compressed(s["cv"]/"predictions.npz",**arrays)
    hash_manifest(s["cv"],"run_manifest.json",s["cv_fields"])
    with pytest.raises(ValueError,match="CV blend formula mismatch"):b.prepare(str(s["config"]))


def test_unsafe_manifest_artifact_rejected(sources):
    s=sources;m=b.read(s["gmm"]/"run_manifest.json");m["artifact_sha256"]["../outside"]= "0"*64;write(s["gmm"]/"run_manifest.json",m)
    with pytest.raises(ValueError,match="Unsafe/missing artifact"):b.prepare(str(s["config"]))


def test_fresh_output_cannot_overlap_source(sources):
    with pytest.raises(ValueError,match="overlaps source"):b.prepare(str(sources["config"]),output=str(sources["gmm"]/"new_blend"))


def test_change_after_preflight_does_not_leave_submission(sources):
    s=sources;c=b.prepare(str(s["config"]))
    (s["gmm"]/"submission.csv").write_text("changed after preflight")
    with pytest.raises(ValueError,match="Changed/missing source"):b.export(c)
    assert not (c["output"]/"submission.csv").exists()
    assert b.read(c["output"]/"prediction_manifest.json")["status"]=="failed_or_interrupted"


def test_non_50_50_design_rejected(sources):
    s=sources;cfg=dict(s["cfg"],design=dict(gmm_weight=.6,trial25_weight=.4));write(s["config"],cfg)
    with pytest.raises(ValueError,match="Fixed50/50"):b.prepare(str(s["config"]))


def test_bad_csv_header_rejected(tmp_path):
    p=tmp_path/"duplicate.csv";p.write_text("id,target,target\n001,0.2,0.3\n")
    with pytest.raises(ValueError,match="duplicate CSV headers"):b.csv_frame(p)


@pytest.mark.parametrize("bad",[np.array([np.nan]),np.array([np.inf]),np.array([1.01]),np.array([-.01])])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):b.probability(bad,1)


def test_cli_dry_run_writes_nothing(sources,monkeypatch,capsys):
    import sys
    s=sources;monkeypatch.setattr(sys,"argv",["blend-export","--config",str(s["config"]),"--dry-run"])
    b.main();result=json.loads(capsys.readouterr().out)
    assert result["fits_performed"]==0 and result["competition_test_opened"] is True
    assert result["output_created"] is False and not Path(result["output"]).exists()


def test_roundtrip_failure_removes_pending_and_final_csv(sources,monkeypatch):
    c=b.prepare(str(sources["config"]));reader=b.csv_frame
    def fail_pending(path,*a,**kw):
        if Path(path).name=="submission.pending.csv":raise ValueError("Injected roundtrip failure")
        return reader(path,*a,**kw)
    monkeypatch.setattr(b,"csv_frame",fail_pending)
    with pytest.raises(ValueError,match="roundtrip failure"):b.export(c)
    assert not (c["output"]/"submission.csv").exists()
    assert not (c["output"]/"submission.pending.csv").exists()
    assert b.read(c["output"]/"prediction_manifest.json")["status"]=="failed_or_interrupted"

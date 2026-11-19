"""One frozen Trial21 refit, two separately identified submission candidates."""
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.submission import generative_trial21_snapshot_v1 as s
from src.features.generative_search_geometry_v1 import SearchGeometryProjector

b=s.b


def predict(bundle,x):
    if list(x.columns)!=bundle["features"]:raise ValueError("Prediction feature order differs")
    z=bundle["projector"].transform(x)
    p=np.mean([m.predict_proba(z)[:,1] for m in bundle["models"]],axis=0)
    b.probability(p,len(x));return p


def prediction_inputs(c):
    cfg=c["export_config"];paths={key:(s.ROOT/cfg[key]).resolve() for key in ["test_path","sample_submission_path"]}
    declared=c["trial25_manifest"]["prediction_input_sha256"]
    # Competition files are first opened here, after all train-only fits and train reload verification.
    hashes={str(p):b.digest(p) for p in paths.values()}
    for h in hashes.values():
        if h not in declared.values():raise ValueError("Trial25 used different test/template contents")
    b.check(declared)
    test=s.csv_frame(paths["test_path"]);template=s.csv_frame(paths["sample_submission_path"],["id","target"])
    features=list(c["x"].columns);n=cfg["expected_rows"]
    if len(test)!=n or len(template)!=n or set(test.id)!=set(template.id):raise ValueError("Test/template row/ID mismatch")
    if set(test.columns)!={"id",*features}:raise ValueError("Test feature schema differs")
    if any(not pd.api.types.is_numeric_dtype(test[col]) or pd.api.types.is_bool_dtype(test[col]) for col in features) or not np.isfinite(test[features].to_numpy(dtype=float)).all():raise ValueError("Invalid test feature values")
    old=s.csv_frame(c["trial25_root"]/"submission.csv",["id","target"]);b.probability(old.target.to_numpy(),n)
    if set(old.id)!=set(template.id):raise ValueError("Trial25 ID set differs")
    audit=s.csv_frame(c["trial25_root"]/"component_predictions.csv")
    if len(audit)!=n or set(audit.id)!=set(template.id) or not {"l1","xgboost","lightgbm","elasticnet","original_three_model","ensemble"}<=set(audit):raise ValueError("Trial25 audit schema/coverage differs")
    for col in ["l1","xgboost","lightgbm","elasticnet","original_three_model","ensemble"]:b.probability(audit[col].to_numpy(),n)
    original=(audit.l1.to_numpy()+audit.xgboost.to_numpy()+audit.lightgbm.to_numpy())/3
    raw=(1-s.TRIAL25_PARAMETERS["alpha"])*original+s.TRIAL25_PARAMETERS["alpha"]*audit.elasticnet.to_numpy()
    if not np.allclose(original,audit.original_three_model,rtol=0,atol=s.TOL) or not np.allclose(raw,audit.ensemble,rtol=0,atol=s.TOL):raise ValueError("Trial25 audit formula differs")
    old_ordered=old.set_index("id").loc[template.id,"target"].to_numpy()
    if not np.allclose(pd.Series(raw,index=audit.id).loc[template.id],old_ordered,rtol=0,atol=s.TOL):raise ValueError("Trial25 audit/submission mismatch")
    b.check(hashes)
    return test,template,old_ordered,hashes


def execute(c):
    out=c["output"];out.mkdir(parents=True,exist_ok=False)
    m=dict(status="running",selected_trial=21,parameters=c["selected_parameters"],started_utc=datetime.now(timezone.utc).isoformat(),planned_candidate_fits=3,
        attempted_candidate_fits=0,completed_candidate_fits=0,completed_projector_fits=0,competition_test_opened=False,competition_test_used_for_fitting=False,
        kaggle_upload_performed=False,automatic_promotion=False,source_input_and_code_sha256=c["hashes"],saved_and_reloaded_models_verified=False)
    finals=[out/"standalone/submission.csv",out/"hybrid/submission.csv"];pending=[p.with_name("submission.pending.csv") for p in finals]
    b.write(out/"prediction_manifest.json",m)
    try:
        b.check(c["hashes"])
        b.write(out/"settings.json",dict(config=c["export_config"],parameters=c["selected_parameters"],seeds=b.DESIGN["initialization_seeds"],covariance="tied",source_design=b.DESIGN,
            versions={p:version(p) for p in ["numpy","scipy","pandas","scikit-learn","joblib","threadpoolctl"]},
            selection="Trial21 selected by HYBRID CV objective. Standalone is a separately reported post-search follow-up, not a new standalone winner selection."))
        b.write(out/"source_decision.json",c["source_decision"]);b.write(out/"source_best_trial.json",c["source_best"]);c["source_summary"].to_csv(out/"source_model_summary.csv",index=False)
        with threadpool_limits(limits=1):
            params=c["selected_parameters"];geometry=SearchGeometryProjector(params["selected_features"],params["latent_dimensions"]).fit(c["x"])
            m["completed_projector_fits"]=1;z=geometry.transform(c["x"]);models=[];traces=[]
            b.write(out/"geometry_diagnostics.json",geometry.search_diagnostics_)
            for seed in b.DESIGN["initialization_seeds"]:
                m["attempted_candidate_fits"]+=1;b.write(out/"prediction_manifest.json",m)
                fitted=s.search.model(params,seed).fit(z,c["y"].to_numpy());m["completed_candidate_fits"]+=1
                traces.append(dict(seed=seed,converged=bool(fitted.converged_),n_iter=int(fitted.n_iter_),initialization_warnings=fitted.initialization_warnings_,objective_trace=fitted.objective_trace_,objective_decreases=int(fitted.objective_decreases_),effective_masses=fitted.effective_masses_,class_probabilities=fitted.class_probabilities_.tolist()))
                b.write(out/"fit_diagnostics.json",traces)
                if not fitted.converged_:raise RuntimeError("Full-data Trial21 EM did not converge; no submissions")
                models.append(fitted);b.write(out/"prediction_manifest.json",m)
            bundle=dict(projector=geometry,models=models,features=list(c["x"].columns),parameters=params,seeds=b.DESIGN["initialization_seeds"],selected_trial=21,mode="training_only")
            joblib.dump(bundle,out/"model_bundle.joblib")
            # Only deserialize this run's freshly written model, never source/untrusted pickle files.
            loaded=joblib.load(out/"model_bundle.joblib")
            np.testing.assert_allclose(predict(bundle,c["x"]),predict(loaded,c["x"]),rtol=0,atol=s.TOL)
            m["competition_test_opened"]=True;b.write(out/"prediction_manifest.json",m)
            test,template,old,inputs=prediction_inputs(c)
            pg=predict(bundle,test[bundle["features"]]);reloaded=predict(loaded,test[bundle["features"]]);np.testing.assert_allclose(pg,reloaded,rtol=0,atol=s.TOL)
            ordered=pd.Series(reloaded,index=test.id).loc[template.id].to_numpy();hybrid=.5*ordered+.5*old
        b.probability(ordered,len(template));b.probability(hybrid,len(template));b.check(c["hashes"]);b.check(inputs)
        audit=pd.DataFrame({"id":template.id.to_numpy(),"trial21_gmm":ordered,"trial25":old,"trial21_hybrid":hybrid})
        audit.to_csv(out/"aligned_predictions.csv",index=False,float_format="%.17g")
        for directory,p,path in [("standalone",ordered,pending[0]),("hybrid",hybrid,pending[1])]:
            path.parent.mkdir(parents=True,exist_ok=False)
            pd.DataFrame({"id":template.id.to_numpy(),"target":p}).to_csv(path,index=False,float_format="%.17g")
            reread=s.csv_frame(path,["id","target"]);b.probability(reread.target.to_numpy(),len(template))
            if not np.array_equal(reread.id.astype(str),template.id.astype(str)) or not np.allclose(reread.target,p,rtol=0,atol=s.TOL):raise ValueError("Submission round-trip failed: "+directory)
        b.check(c["hashes"]);b.check(inputs)
        if m["completed_candidate_fits"]!=3:raise ValueError("Fit accounting differs")
        for p,q in zip(pending,finals):p.replace(q)
        for name,p,role,formula in [("standalone",finals[0],"post_search_standalone_followup","mean_of_three_Trial21_GMM_seeds"),("hybrid",finals[1],"original_search_objective_candidate","0.5*p_Trial21_GMM+0.5*p_frozen_Trial25")]:
            b.write(p.parent/"candidate_manifest.json",dict(candidate="generative_trial21_"+name,role=role,formula=formula,rows=len(template),submission_sha256=b.digest(p),model_bundle_sha256=b.digest(out/"model_bundle.joblib"),source_trial25_submission_sha256=b.digest(c["trial25_root"]/"submission.csv"),automatic_upload=False))
        m.update(status="ready_for_manual_submission",rows=len(template),prediction_input_sha256=inputs,saved_and_reloaded_models_verified=True,csv_roundtrip_verified=True,
            submission_sha256={str(p.relative_to(out)):b.digest(p) for p in finals})
        m["artifact_sha256"]={str(p.relative_to(out)):b.digest(p) for p in out.rglob("*") if p.is_file() and p.name!="prediction_manifest.json"}
        return dict(status=m["status"],output=str(out),fits_performed=3,rows=len(template),candidates=["standalone","hybrid"],automatic_upload=False)
    except BaseException as exc:
        for p in finals+pending:p.unlink(missing_ok=True)
        for p in [out/"standalone/candidate_manifest.json",out/"hybrid/candidate_manifest.json"]:p.unlink(missing_ok=True)
        m.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        m["finished_utc"]=datetime.now(timezone.utc).isoformat();b.write(out/"prediction_manifest.json",m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/generative_trial21_export_v1.json");p.add_argument("--output");p.add_argument("--dry-run",action="store_true")
    args=p.parse_args();c=s.prepare(args.config,args.output)
    if args.dry_run:print(json.dumps(dict(status="preflight_passed",selected_trial=21,planned_candidate_fits=3,fits_performed=0,competition_test_opened=False,prediction_input_validation="deferred_until_after_fitting",output_created=False,output=str(c["output"])),indent=2))
    else:print(json.dumps(execute(c),indent=2))

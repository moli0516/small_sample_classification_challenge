"""Frozen A_k8_tied refit/export. Competition features are prediction-only."""
import csv
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.experiments import generative_geometry_v1 as base
from src.experiments import generative_followup_v1 as f


def bundle_predict(bundle,x):
    if list(x.columns)!=bundle["features"]:raise ValueError("Model feature order differs")
    z=bundle["projector"].transform(x)
    p=np.mean([m.predict_proba(z)[:,1] for m in bundle["models"]],axis=0)
    base.probability(p,len(x));return p


def read_csv(path,id_column):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:header=next(csv.reader(stream),[])
    if not header or len(set(header))!=len(header) or any(not s.strip() for s in header):raise ValueError("Invalid CSV headers")
    frame=pd.read_csv(path,dtype={id_column:"string"})
    if id_column not in frame or frame[id_column].isna().any() or not frame[id_column].is_unique or frame[id_column].str.strip().eq("").any():raise ValueError("Invalid/duplicate IDs")
    return frame


def align_submission(test,template,features,p,expected_rows,id_column="id",target_column="target"):
    if len(test)!=expected_rows or len(template)!=expected_rows:raise ValueError("Wrong test/template row count")
    if list(template.columns)!=[id_column,target_column]:raise ValueError("Template must have id,target columns in that order")
    if set(test.columns)!={id_column,*features}:raise ValueError("Test feature schema mismatch")
    for frame in [test,template]:
        if frame[id_column].isna().any() or not frame[id_column].is_unique:raise ValueError("Invalid IDs")
    if set(test[id_column])!=set(template[id_column]):raise ValueError("Test/template ID sets differ")
    base.probability(p,len(test))
    values=pd.Series(p,index=test[id_column]).loc[template[id_column]].to_numpy()
    return pd.DataFrame({id_column:template[id_column].to_numpy(),target_column:values})


def export(c):
    out=c["output"];out.mkdir(parents=True,exist_ok=False);m=f.manifest(c);m["planned_candidate_fits"]=3
    m["saved_and_reloaded_models_verified"]=False;base.write(out/"run_manifest.json",m)
    try:
        base.check(c["hashes"]);base.write(out/"settings.json",f.settings(c))
        with threadpool_limits(limits=1):
            p=f.projector("geometry").fit(c["x"]);m["completed_projector_fits"]+=1
            z=p.transform(c["x"]);models=[];traces=[]
            for seed in base.DESIGN["initialization_seeds"]:
                fitted=f.model(8,seed).fit(z,c["y"].to_numpy())
                traces.append(dict(seed=seed,**f.fit_trace(fitted)));base.write(out/"fit_diagnostics.json",traces)
                if not fitted.converged_:raise RuntimeError("Full-data EM did not converge; no submission exported")
                models.append(fitted);m["completed_candidate_fits"]+=1;base.write(out/"run_manifest.json",m)
            bundle=dict(projector=p,models=models,features=c["features"],seeds=base.DESIGN["initialization_seeds"],k=8,covariance="tied",mode="A")
            joblib.dump(bundle,out/"model_bundle.joblib")
            # Only load this run's freshly written local model, never arbitrary external pickle files.
            loaded=joblib.load(out/"model_bundle.joblib")
            np.testing.assert_allclose(bundle_predict(bundle,c["x"]),bundle_predict(loaded,c["x"]),rtol=0,atol=base.TOL)
            # Test features are first opened AFTER all three train-only fits and train reload check.
            cfg=c["followup_config"];testpath=(f.ROOT/cfg["test_path"]).resolve();templatepath=(f.ROOT/cfg["sample_submission_path"]).resolve()
            m["competition_test_opened"]=True;base.write(out/"run_manifest.json",m)
            hashes={str(testpath):base.digest(testpath),str(templatepath):base.digest(templatepath)}
            test=read_csv(testpath,c["config"]["id_column"]);template=read_csv(templatepath,c["config"]["id_column"])
            idcol=c["config"]["id_column"];target=c["config"]["target_column"]
            if set(test.columns)!={idcol,*c["features"]}:raise ValueError("Test feature schema mismatch")
            if any(not pd.api.types.is_numeric_dtype(test[col]) or pd.api.types.is_bool_dtype(test[col]) for col in c["features"]):raise ValueError("Nonnumeric test features")
            x=test[c["features"]]
            if not np.isfinite(x.to_numpy(dtype=float)).all():raise ValueError("Nonfinite test features")
            predicted=bundle_predict(bundle,x);reloaded=bundle_predict(loaded,x)
            np.testing.assert_allclose(predicted,reloaded,rtol=0,atol=base.TOL)
            submission=align_submission(test,template,c["features"],reloaded,cfg["expected_test_rows"],idcol,target)
        base.check(c["hashes"]);base.check(hashes)
        submission.to_csv(out/"submission.csv",index=False,float_format="%.17g")
        saved=pd.read_csv(out/"submission.csv",dtype={idcol:"string"},float_precision="round_trip")
        if list(saved.columns)!=[idcol,target] or not np.array_equal(saved[idcol].astype(str),template[idcol].astype(str)):raise ValueError("Saved submission identity mismatch")
        np.testing.assert_allclose(saved[target],submission[target],rtol=0,atol=base.TOL)
        m.update(status="ready_for_manual_submission",rows=len(saved),saved_and_reloaded_models_verified=True,export_input_sha256=hashes)
        return dict(status=m["status"],rows=len(saved),output=str(out),candidate=f.CONTROL,fits_performed=m["completed_candidate_fits"],automatic_upload=False)
    except BaseException as exc:
        m.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
        (out/"submission.csv").unlink(missing_ok=True);raise
    finally:f.finish(c,m)

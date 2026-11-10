"""Bounded elastic-net logistic ensemble experiment. No CatBoost retraining."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from inspect import signature
from itertools import product
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
TOL=1e-12
CS=[.1,1.,10.]
RATIOS=[.25,.5,.75]
ALPHAS=[.05,.1,.25]


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x


def write(path,data):Path(path).write_text(json.dumps(safe(data),indent=2,allow_nan=False)+"\n",encoding="utf-8")


def check_hashes(hashes):
    for path,h in hashes.items():
        if digest(path)!=h:raise ValueError("Source/code/input changed: "+path)


def probability(p,n):
    p=np.asarray(p,dtype=float)
    if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("Invalid probability")
    return p


def model_settings():
    return [dict(C=c,l1_ratio=r) for c,r in product(CS,RATIOS)]


def build_model(spec):
    params=dict(C=spec["C"],l1_ratio=spec["l1_ratio"],solver="saga",max_iter=20000,tol=1e-4,random_state=42)
    penalty=signature(LogisticRegression).parameters.get("penalty")
    if penalty is not None and penalty.default!="deprecated":params["penalty"]="elasticnet"
    return Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(**params))])


def compose(base,baseline,raw,tag,spec):
    raw=probability(raw,len(base));preds={};specs={}
    replaced=np.asarray(base,dtype=float).copy();replaced[:,0]=raw
    name=tag+"_replace_l1"
    preds[name]=replaced.mean(axis=1);specs[name]=dict(spec,mode="replace_l1",weights=[1/3,1/3,1/3])
    for alpha in ALPHAS:
        name=tag+f"_add_alpha{alpha:g}"
        preds[name]=(1-alpha)*baseline+alpha*raw
        specs[name]=dict(spec,mode="add_to_original",alpha=alpha)
    return preds,specs


def score(c,preds):
    rows=[]
    for name,p in preds.items():
        p=probability(p,len(c["frame"]))
        for s in c["groups"]:
            sl=slice(s["start"],s["stop"])
            rows.append(dict(candidate=name,repeat=s["repeat"],fold=s["fold"],auc=float(roc_auc_score(c["frame"].target.iloc[sl],p[sl]))))
    folds=pd.DataFrame(rows)
    repeats=folds.groupby(["candidate","repeat"],as_index=False).auc.mean().rename(columns={"auc":"mean_fold_auc"})
    summary=repeats.groupby("candidate").mean_fold_auc.agg(mean_auc="mean",repeat_sd="std").reset_index()
    return folds,repeats,summary


def compare(repeats,names,reference):
    def get(name):return repeats[repeats.candidate==name].set_index("repeat").mean_fold_auc.sort_index()
    first=get(reference)
    if list(first.index)!=list(range(10)):raise ValueError("Ten repetitions required")
    details=[];stats=[]
    for name in names:
        second=get(name)
        if not second.index.equals(first.index):raise ValueError("Unpaired scores")
        delta=second-first
        details.extend(dict(reference=reference,candidate=name,repeat=int(r),difference=float(d)) for r,d in delta.items())
        stats.append(dict(reference=reference,candidate=name,mean_difference=float(delta.mean()),difference_sd=float(delta.std(ddof=1)),wins=int((delta>TOL).sum()),losses=int((delta< -TOL).sum()),ties=int((abs(delta)<=TOL).sum())))
    return pd.DataFrame(details),pd.DataFrame(stats)


def prepare(config_path,output=None):
    from scripts import run_ensemble_research_suite_v1 as suite
    c=suite.prepare(config_path)
    dest=(ROOT/(output or str(c["output_root"]/"elasticnet_v1"))).resolve()
    if dest.exists():raise FileExistsError("Fresh output required; no overwrite/resume")
    source=c["output_root"]/"catboost"
    if source.is_relative_to(dest) or dest.is_relative_to(source):raise ValueError("Output overlaps CatBoost source")
    decision,z,hashes=suite.load_child(c,"catboost")
    if decision["selected_candidate"]!="cat_10_alpha0.1":raise ValueError("Expected frozen CatBoost challenger")
    control=probability(z["prediction__"+decision["selected_candidate"]],len(c["frame"]))
    _,_,summary=score(c,{"catboost_frozen":control})
    if not np.isclose(float(summary.mean_auc.iloc[0]),decision["selected_mean_auc"],rtol=0,atol=TOL):raise ValueError("CatBoost challenger score did not reproduce")
    watched=dict(c["hashes"]);watched.update(hashes);watched[str(Path(__file__).resolve())]=digest(__file__)
    check_hashes(watched)
    c.update(output=dest,catboost_control=control,elasticnet_hashes=watched)
    return c


def execute(c,model_builder=build_model):
    out=c["output"];out.mkdir(parents=True,exist_ok=False)
    specs0=model_settings();n=len(c["frame"])
    raw={f"en_C{s['C']:g}_r{s['l1_ratio']:g}":np.empty(n) for s in specs0}
    manifest=dict(status="running",planned_fits=len(specs0)*len(c["groups"]),completed_fits=0,test_opened=False,kaggle_upload_performed=False,started_utc=datetime.now(timezone.utc).isoformat())
    warnrows=[];diagnostics=[]
    write(out/"run_manifest.json",manifest)
    try:
        write(out/"settings.json",dict(grid=specs0,alphas=ALPHAS,solver="saga",max_iter=20000,tol=1e-4,random_state=42,
              input_and_code_sha256=c["elasticnet_hashes"],versions={p:version(p) for p in ["numpy","pandas","scikit-learn"]},
              objective="Highest ensemble mean over 9 replacement and 27 addition candidates; standalone scores are diagnostic only. No CatBoost added to these candidates; frozen CatBoost blend is a second control."))
        c["frame"].to_csv(out/"cohort.csv",index=False);write(out/"split_indices.json",c["groups"])
        with threadpool_limits(limits=1):
            for s in c["groups"]:
                tr,va=s["train_indices"],s["valid_indices"]
                for spec in specs0:
                    tag=f"en_C{spec['C']:g}_r{spec['l1_ratio']:g}";model=model_builder(spec)
                    caught=[]
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(c["context"]["x"].iloc[tr],c["context"]["y"].iloc[tr])
                    finally:
                        warnrows.extend(dict(repeat=s["repeat"],fold=s["fold"],model=tag,category=w.category.__name__,message=str(w.message)) for w in caught)
                        write(out/"warnings.json",warnrows)
                    if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError("ConvergenceWarning; do not select partial results")
                    p=model.predict_proba(c["context"]["x"].iloc[va])[:,1]
                    raw[tag][s["start"]:s["stop"]]=probability(p,len(va))
                    classifier=model.named_steps["classifier"];coef=classifier.coef_[0]
                    diagnostics.append(dict(repeat=s["repeat"],fold=s["fold"],candidate=tag,nonzero_coefficients=int(np.count_nonzero(coef)),
                         selected_features=[str(col) for col,v in zip(c["context"]["x"].columns,coef) if v!=0],n_iter=int(classifier.n_iter_.max())))
                    manifest["completed_fits"]+=1
                write(out/"run_manifest.json",manifest)
                print(f"[{manifest['completed_fits']}/{manifest['planned_fits']}] repeat={s['repeat']} fold={s['fold']}",flush=True)
        check_hashes(c["elasticnet_hashes"])
        candidates={};candidate_specs={}
        for spec in specs0:
            tag=f"en_C{spec['C']:g}_r{spec['l1_ratio']:g}"
            preds,ss=compose(c["base"],c["baseline"],raw[tag],tag,spec)
            candidates.update(preds);candidate_specs.update(ss)
        all_preds={"reference":c["baseline"],"catboost_frozen":c["catboost_control"],**candidates}
        folds,repeats,summary=score(c,all_preds)
        paired,comparison=compare(repeats,list(candidates),"reference")
        paired_cat,comparison_cat=compare(repeats,list(candidates),"catboost_frozen")
        means=summary.set_index("candidate").mean_auc
        selected=max(candidate_specs,key=lambda name:means[name])
        stat=comparison[comparison.candidate==selected].iloc[0].to_dict()
        catstat=comparison_cat[comparison_cat.candidate==selected].iloc[0].to_dict()
        passed=lambda s:bool(s["mean_difference"]>TOL and s["wins"]>=6)
        decision=dict(selected_candidate=selected,selected_mean_auc=float(means[selected]),selected_spec=candidate_specs[selected],
              screening_passed=passed(stat),catboost_control_comparison=catstat,screening_both_controls=bool(passed(stat) and passed(catstat)),
              automatic_promotion=False,selection_rule="Highest mean of all36 ensembles; exact ties use C ascending, ratio ascending, replacement before alpha ascending; no fallback.",
              interpretation="Adaptive development comparison on reused splits; no independent evaluation or significance test.",**stat)
        for name,df in [("fold_scores",folds),("repeat_scores",repeats),("model_summary",summary),("comparison",comparison),
                        ("comparison_vs_catboost",comparison_cat),("paired_repeat_differences",paired),("paired_vs_catboost",paired_cat)]:df.to_csv(out/(name+".csv"),index=False)
        _,_,standalone=score(c,raw);standalone.to_csv(out/"standalone_summary.csv",index=False)
        write(out/"coefficient_diagnostics.json",diagnostics)
        write(out/"candidate_specs.json",candidate_specs);write(out/"decision.json",decision)
        np.savez_compressed(out/"predictions.npz",**all_preds,**{"raw__"+name:p for name,p in raw.items()})
        check_hashes(c["elasticnet_hashes"])
        manifest["status"]="completed"
        manifest["artifact_sha256"]={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!="run_manifest.json"}
        return dict(status="completed",completed_fits=manifest["completed_fits"],output=str(out),automatic_promotion=False)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_utc"]=datetime.now(timezone.utc).isoformat();write(out/"run_manifest.json",manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/ensemble_research_suite_v1.json")
    p.add_argument("--output")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args();c=prepare(args.config,args.output)
    if args.dry_run:
        print(json.dumps(dict(status="preflight_passed",models=9,ensemble_candidates=36,planned_fits=450,fits_performed=0,test_opened=False,output=str(c["output"])),indent=2))
    else:print(json.dumps(execute(c),indent=2))


if __name__=="__main__":main()

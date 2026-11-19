"""Independent generative follow-ups; frozen K8 export never consumes search winners."""
import argparse
from datetime import datetime,timezone
from itertools import product
from importlib.metadata import version
import json
from pathlib import Path
import inspect
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from src.experiments import generative_geometry_v1 as base
from src.features.generative_geometry_v1 import GeometryProjector
from src.features.generative_followup_v1 import PlainPCAProjector
from src.modeling.semi_supervised_mixture_v1 import SemiSupervisedMixture

ROOT=Path(__file__).resolve().parents[2]
CONTROL="A_k8_tied"
EXPECTED_AUC=.9296401907271473
DESIGN=dict(capacity_k=[8,12,16],ablation=["geometry","plain_pca"],blend_gmm_weights=[.25,.5,.75],seed_policy="average_all_fixed_seeds")


def verify_artifacts(root):
    root=Path(root).resolve();manifest=base.read(root/"run_manifest.json")
    needed={"settings.json","decision.json","cohort.csv","split_indices.json","predictions.npz","candidate_specs.json"}
    hashes=manifest.get("artifact_sha256",{})
    if manifest.get("status")!="completed" or not needed<=hashes.keys():raise ValueError("Completed generative study with required artifact hashes needed")
    if manifest.get("completed_candidate_fits")!=manifest.get("planned_candidate_fits"):raise ValueError("Incomplete generative fit budget")
    result={str(root/"run_manifest.json"):base.digest(root/"run_manifest.json")}
    for name,h in hashes.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or base.digest(p)!=h:raise ValueError("Unsafe/changed generative artifact: "+name)
        result[str(p)]=h
    return result


def prepare(config_path,action,output=None,source_study=None):
    config_path=(ROOT/config_path).resolve();cfg=base.read(config_path)
    required={"source_config","source_study","output_root","submission_output","test_path","sample_submission_path","expected_test_rows","design"}
    if set(cfg)!=required or cfg["design"]!=DESIGN:raise ValueError("Frozen follow-up design required")
    if action not in ("export","capacity","ablation","blend"):raise ValueError("Unknown action")
    source=(ROOT/(source_study or cfg["source_study"])).resolve()
    dest=(ROOT/(output or (cfg["submission_output"] if action=="export" else str(Path(cfg["output_root"])/action)))).resolve()
    if dest.exists():raise FileExistsError("Fresh output required; no overwrite/resume")
    if dest.is_relative_to(source) or source.is_relative_to(dest):raise ValueError("Output overlaps generative source")
    c=base.prepare(str((ROOT/cfg["source_config"]).resolve()),output=str(dest))
    hashes=dict(c["hashes"]);hashes.update(verify_artifacts(source))
    settings=base.read(source/"settings.json")
    if settings["config"]!=c["config"] or "A" not in settings["modes"]:raise ValueError("Source config/scope mismatch")
    for pkg,expected in settings["versions"].items():
        if version(pkg)!=expected:raise ValueError("Source environment version changed: "+pkg)
    base.check(settings["input_and_code_sha256"]);hashes.update(settings["input_and_code_sha256"])
    actual=pd.read_csv(source/"cohort.csv",dtype={"id":"string"},float_precision="round_trip")
    expected=c["frame"].copy();expected["id"]=expected.id.astype("string")
    if not actual.equals(expected) or base.read(source/"split_indices.json")!=c["groups"]:raise ValueError("Generative cohort/splits differ")
    decision=base.read(source/"decision.json")
    if decision.get("selected_candidate")!=CONTROL or not decision.get("screening_passed") or not decision.get("eligible_candidates",{}).get(CONTROL):raise ValueError("Expected eligible screening-passed A_k8_tied winner")
    specs=base.read(source/"candidate_specs.json")
    if specs.get(CONTROL)!=dict(mode="A",n_components=8,covariance_type="tied"):raise ValueError("Control specification mismatch")
    with np.load(source/"predictions.npz",allow_pickle=False) as z:
        control=np.asarray(z[CONTROL],dtype=float);reference=np.asarray(z["reference_trial25"],dtype=float)
    base.probability(control,len(c["frame"]));base.probability(reference,len(c["frame"]))
    if not np.allclose(reference,c["reference"],rtol=0,atol=base.TOL):raise ValueError("Trial25 reference prediction mismatch")
    _,_,summary=base.evaluate(c,{CONTROL:control});auc=float(summary.mean_auc.iloc[0])
    if not np.isclose(auc,EXPECTED_AUC,rtol=0,atol=base.TOL) or not np.isclose(auc,decision["selected_mean_auc"],rtol=0,atol=base.TOL):raise ValueError("K8 control AUC did not reproduce")
    from src.features import generative_followup_v1
    from src.submission import generative_export_v1
    for p in [config_path,Path(__file__).resolve(),Path(inspect.getfile(generative_followup_v1)),Path(inspect.getfile(generative_export_v1)),ROOT/"scripts/run_generative_followup_v1.py"]:
        hashes[str(p)]=base.digest(p)
    base.check(hashes)
    if action=="export":
        for key in ["test_path","sample_submission_path"]:
            p=(ROOT/cfg[key]).resolve()
            if not p.is_file():raise FileNotFoundError(f"{key}: {p}; correct this path in the follow-up config")
            if p.is_relative_to(dest):raise ValueError("Export output overlaps input")
    c.update(output=dest,hashes=hashes,followup_config=cfg,control=control,action=action,source_study=str(source))
    return c


def model(k,seed):
    d=base.DESIGN
    return SemiSupervisedMixture(n_components=k,covariance_type="tied",unlabelled_weight=d["unlabelled_weight"],
        reg_covar=d["reg_covar"],label_smoothing=d["label_smoothing"],max_iter=d["max_iter"],tol=d["tol"],random_state=seed)


def projector(kind):
    d=base.DESIGN
    return GeometryProjector(d["selected_features"],d["latent_dimensions"]) if kind=="geometry" else PlainPCAProjector(d["latent_dimensions"])


def fit_trace(m):
    return dict(converged=bool(m.converged_),n_iter=int(m.n_iter_),initialization_warnings=m.initialization_warnings_,
        objective_trace=m.objective_trace_,objective_decreases=int(m.objective_decreases_),effective_masses=m.effective_masses_,class_probabilities=m.class_probabilities_.tolist())


def manifest(c):
    return dict(status="running",action=c["action"],started_utc=datetime.now(timezone.utc).isoformat(),competition_test_opened=False,
        competition_test_used_for_fitting=False,kaggle_upload_performed=False,completed_candidate_fits=0,completed_projector_fits=0,
        fit_accounting="One candidate fit includes one sklearn GMM initialization and one custom EM run")


def finish(c,m):
    m["finished_utc"]=datetime.now(timezone.utc).isoformat()
    if m["status"] in ("completed","ready_for_manual_submission"):
        m["artifact_sha256"]={str(p.relative_to(c["output"])):base.digest(p) for p in c["output"].rglob("*") if p.is_file() and p.name!="run_manifest.json"}
    base.write(c["output"]/"run_manifest.json",m)


def settings(c):
    return dict(config=c["followup_config"],source_study=c["source_study"],source_design=base.DESIGN,input_and_code_sha256=c["hashes"],
        versions={p:version(p) for p in ["numpy","scipy","pandas","scikit-learn","threadpoolctl","joblib"]},
        scope="Training-only fits; fixed K8 control. Export does not consume follow-up winners. Reused-label exploratory development, not independent CV.")


def run_research(c):
    out=c["output"];out.mkdir(parents=True,exist_ok=False);m=manifest(c)
    action=c["action"];new={}
    if action=="capacity":new={f"A_k{k}_tied":dict(k=k,representation="geometry") for k in DESIGN["capacity_k"] if k!=8}
    elif action=="ablation":new={"A_k8_plain_pca_tied":dict(k=8,representation="plain_pca")}
    elif action!="blend":raise ValueError("Expected research action")
    m["planned_candidate_fits"]=len(new)*len(c["groups"])*len(base.DESIGN["initialization_seeds"])
    base.write(out/"run_manifest.json",m)
    predictions={CONTROL:c["control"],"reference_trial25":c["reference"]};eligible={CONTROL:True}
    for name in new:predictions[name]=np.full(len(c["frame"]),np.nan);eligible[name]=True
    try:
        base.check(c["hashes"]);base.write(out/"settings.json",settings(c))
        c["frame"].to_csv(out/"cohort.csv",index=False);base.write(out/"split_indices.json",c["groups"])
        with threadpool_limits(limits=1),(out/"fit_diagnostics.jsonl").open("w") as stream:
            for s in c["groups"] if new else []:
                tr,va=s["train_indices"],s["valid_indices"]
                for name,spec in new.items():
                    p=projector(spec["representation"]).fit(c["x"].iloc[tr]);m["completed_projector_fits"]+=1
                    zl,zv=p.transform(c["x"].iloc[tr]),p.transform(c["x"].iloc[va]);ps=[]
                    for seed in base.DESIGN["initialization_seeds"]:
                        fitted=model(spec["k"],seed).fit(zl,c["y"].iloc[tr].to_numpy())
                        raw=fitted.predict_proba(zv)[:,1];base.probability(raw,len(va));ps.append(raw)
                        eligible[name]=eligible[name] and bool(fitted.converged_)
                        stream.write(json.dumps(dict(candidate=name,repeat=s["repeat"],fold=s["fold"],seed=seed,**fit_trace(fitted)),allow_nan=False)+"\n");stream.flush()
                        m["completed_candidate_fits"]+=1
                    predictions[name][s["start"]:s["stop"]]=np.mean(ps,axis=0)
                base.write(out/"run_manifest.json",m)
                print(f"{action}: repeat={s['repeat']} fold={s['fold']} fits={m['completed_candidate_fits']}/{m['planned_candidate_fits']}",flush=True)
        candidate_specs={CONTROL:dict(k=8,representation="geometry",source="frozen_original_study"),**new}
        candidates=[CONTROL,*new]
        if action=="blend":
            candidates=[]
            for weight in DESIGN["blend_gmm_weights"]:
                name=f"blend_gmm{weight:g}";candidates.append(name)
                predictions[name]=weight*c["control"]+(1-weight)*c["reference"];eligible[name]=True
                candidate_specs[name]=dict(gmm_weight=weight,trial25_weight=1-weight)
        folds,repeats,summary=base.evaluate(c,predictions)
        comparisons=[base.compare(repeats,name,ref) for name in candidates for ref in [CONTROL,"reference_trial25"] if name!=ref]
        means=summary.set_index("candidate").mean_auc;valid=[n for n in candidates if eligible[n]]
        selected=max(valid,key=lambda n:means[n]) if valid else None
        checks=[v for v in comparisons if v["candidate"]==selected]
        passed=lambda v:v["mean_difference"]>base.TOL and v["wins"]>=6
        decision=dict(action=action,selected_candidate=selected,selected_mean_auc=float(means[selected]) if selected else None,selected_spec=candidate_specs.get(selected),eligible_candidates=eligible,
            comparisons=checks,screening_against_both_controls=bool(len(checks)==2 and all(passed(v) for v in checks)),automatic_promotion=False,
            selection_rule="Highest eligible mean; exact ties generation order. Capacity/ablation include frozen K8. Blending selects only its three fixed weights. No seed selection.",
            interpretation="Exploratory reused-split evidence; follow-up cannot change the frozen K8 export recipe")
        for name,df in [("fold_scores",folds),("repeat_scores",repeats),("model_summary",summary),("comparison",pd.DataFrame(comparisons))]:df.to_csv(out/(name+".csv"),index=False)
        if action=="blend":
            # Threshold diagnostics are not selection metrics; primary AUC remains mean-fold then mean-repeat.
            rows=[]
            for s in c["groups"]:
                sl=slice(s["start"],s["stop"]);a=c["control"][sl];b=c["reference"][sl];y=c["frame"].target.iloc[sl].to_numpy()
                corr=spearmanr(a,b).statistic
                rows.append(dict(repeat=s["repeat"],fold=s["fold"],prediction_spearman=float(corr) if np.isfinite(corr) else None,
                    threshold_disagreement=float(np.mean((a>=.5)!=(b>=.5))),threshold_both_wrong=float(np.mean(((a>=.5)!=y)&((b>=.5)!=y)))))
            pd.DataFrame(rows).to_csv(out/"complementarity.csv",index=False)
        base.write(out/"candidate_specs.json",candidate_specs)
        base.write(out/"decision.json",decision);np.savez_compressed(out/"predictions.npz",**predictions)
        base.check(c["hashes"]);m["status"]="completed";return decision
    except BaseException as exc:
        m.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}");raise
    finally:finish(c,m)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["export","capacity","ablation","blend"])
    p.add_argument("--config",default="configs/generative_followup_v1.json")
    p.add_argument("--source-study");p.add_argument("--output");p.add_argument("--dry-run",action="store_true")
    args=p.parse_args();c=prepare(args.config,args.action,args.output,args.source_study)
    if args.dry_run:
        budgets=dict(export=3,capacity=300,ablation=150,blend=0)
        print(json.dumps(dict(status="preflight_passed",action=args.action,planned_candidate_fits=budgets[args.action],fits_performed=0,
            competition_test_opened=False,output=str(c["output"]),frozen_candidate=CONTROL),indent=2))
    elif args.action=="export":
        from src.submission.generative_export_v1 import export
        print(json.dumps(export(c),indent=2))
    else:print(json.dumps(run_research(c),indent=2))

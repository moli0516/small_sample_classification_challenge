"""Bounded native Optuna search of the training-only generative branch."""
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.experiments import generative_optuna_snapshot_v1 as snapshot
from src.features.generative_search_geometry_v1 import SearchGeometryProjector
from src.modeling.semi_supervised_mixture_v1 import SemiSupervisedMixture

b=snapshot.frozen.base
CONTROL=dict(selected_features=20,latent_dimensions=8,n_components=8,reg_covar=.001,label_smoothing=.001)


class RejectedTrial(RuntimeError):pass
class ControlFailure(RuntimeError):pass


def suggest(trial):
    d=snapshot.DESIGN
    p={k:trial.suggest_categorical(k,d[k]) for k in ["selected_features","latent_dimensions","n_components"]}
    p.update({k:trial.suggest_float(k,*d[k],log=True) for k in ["reg_covar","label_smoothing"]})
    return p


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def dump(p,value):b.write(p,safe(value))


def model(params,seed):
    d=b.DESIGN
    return SemiSupervisedMixture(n_components=params["n_components"],covariance_type="tied",unlabelled_weight=d["unlabelled_weight"],
        reg_covar=params["reg_covar"],label_smoothing=params["label_smoothing"],max_iter=d["max_iter"],tol=d["tol"],random_state=seed)


def numerical_value_error(exc):
    text=str(exc).lower()
    return "ill-defined empirical covariance" in " ".join(text.split())


def fit_trial(c,trial,params,stats):
    directory=c["output"]/"trials"/f"trial_{trial.number:04d}";directory.mkdir(parents=True,exist_ok=False)
    dump(directory/"parameters.json",params);raw=np.full(len(c["frame"]),np.nan)
    try:
        with threadpool_limits(limits=1),(directory/"fit_diagnostics.jsonl").open("w",encoding="utf-8") as log,(directory/"geometry_diagnostics.jsonl").open("w",encoding="utf-8") as geometries:
            for s in c["groups"]:
                b.check(c["hashes"]);tr,va=s["train_indices"],s["valid_indices"]
                stats["attempted_projector_fits"]+=1
                with np.errstate(over="raise",divide="raise",invalid="raise"):
                    geometry=SearchGeometryProjector(params["selected_features"],params["latent_dimensions"]).fit(c["x"].iloc[tr])
                    zl=geometry.transform(c["x"].iloc[tr]);zv=geometry.transform(c["x"].iloc[va])
                if not np.isfinite(zl).all() or not np.isfinite(zv).all():raise FloatingPointError("Nonfinite transformed features")
                stats["completed_projector_fits"]+=1
                geometries.write(json.dumps(safe(dict(repeat=s["repeat"],fold=s["fold"],**geometry.search_diagnostics_)),allow_nan=False)+"\n");geometries.flush()
                predictions=[]
                for seed in b.DESIGN["initialization_seeds"]:
                    fitted=model(params,seed);stats["attempted_candidate_fits"]+=1;caught=[];error=None
                    try:
                        with warnings.catch_warnings(record=True) as caught,np.errstate(over="raise",divide="raise",invalid="raise"):
                            warnings.simplefilter("always")
                            fitted.fit(zl,c["y"].iloc[tr].to_numpy())
                            stats["completed_candidate_fits"]+=1
                            if not fitted.converged_:raise RejectedTrial("Custom EM did not converge")
                            stats["converged_candidate_fits"]+=1
                            p=fitted.predict_proba(zv)[:,1]
                            if not np.isfinite(p).all():raise FloatingPointError("Nonfinite predicted probabilities")
                            b.probability(p,len(va));predictions.append(p)
                    except BaseException as exc:
                        error=f"{type(exc).__name__}: {exc}";raise
                    finally:
                        record=dict(repeat=s["repeat"],fold=s["fold"],seed=seed,error=error,
                            warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
                        for key in ["converged_","n_iter_","initialization_warnings_","objective_trace_","objective_decreases_","effective_masses_","class_probabilities_"]:
                            record[key.rstrip("_")]=getattr(fitted,key,None)
                        log.write(json.dumps(safe(record),allow_nan=False)+"\n");log.flush()
                        dump(c["output"]/"run_manifest.json",stats)
                raw[s["start"]:s["stop"]]=np.mean(predictions,axis=0)
                print(f"trial={trial.number} repeat={s['repeat']} fold={s['fold']} fits={stats['completed_candidate_fits']}/{stats['planned_candidate_fits']}",flush=True)
        hybrid=.5*raw+.5*c["reference"]
        predictions=dict(candidate_gmm=raw,candidate_hybrid=hybrid,original_gmm=c["control"],original_hybrid=c["hybrid_control"],reference_trial25=c["reference"])
        folds,repeats,summary=b.evaluate(c,predictions);means=summary.set_index("candidate").mean_auc
        comparisons=[b.compare(repeats,"candidate_hybrid",name) for name in ["original_hybrid","original_gmm","reference_trial25"]]
        comparisons.append(b.compare(repeats,"candidate_gmm","original_gmm"))
        if trial.number==0:
            if params!=CONTROL:raise ControlFailure("Trial0 parameter identity changed")
            if not np.allclose(raw,c["control"],rtol=0,atol=snapshot.TOL) or not np.allclose(hybrid,c["hybrid_control"],rtol=0,atol=snapshot.TOL):raise ControlFailure("Trial0 row-level predictions did not reproduce")
            if not np.isclose(means["candidate_gmm"],snapshot.GMM_AUC,rtol=0,atol=snapshot.TOL) or not np.isclose(means["candidate_hybrid"],snapshot.HYBRID_AUC,rtol=0,atol=snapshot.TOL):raise ControlFailure("Trial0 AUC did not reproduce")
        for name,df in [("fold_scores",folds),("repeat_scores",repeats),("model_summary",summary),("paired_comparisons",pd.DataFrame(comparisons))]:df.to_csv(directory/(name+".csv"),index=False)
        differences=[]
        for current,reference in [("candidate_hybrid","original_hybrid"),("candidate_hybrid","original_gmm"),("candidate_hybrid","reference_trial25"),("candidate_gmm","original_gmm")]:
            a=repeats[repeats.candidate==current].set_index("repeat").mean_fold_auc
            z=repeats[repeats.candidate==reference].set_index("repeat").mean_fold_auc
            differences.extend(dict(candidate=current,reference=reference,repeat=int(r),difference=float(v)) for r,v in (a-z).items())
        pd.DataFrame(differences).to_csv(directory/"paired_repeat_differences.csv",index=False)
        result=dict(number=trial.number,state="COMPLETE",parameters=params,objective_auc=float(means["candidate_hybrid"]),standalone_auc=float(means["candidate_gmm"]),comparisons=comparisons,control_reproduced=True if trial.number==0 else None)
        np.savez_compressed(directory/"predictions.npz",gmm=raw,hybrid=hybrid)
        dump(directory/"result.json",result);b.check(c["hashes"])
        trial.set_user_attr("standalone_auc",result["standalone_auc"])
        if trial.number==0:trial.set_user_attr("control_reproduced",True)
        return result["objective_auc"]
    except BaseException as exc:
        dump(directory/"error.json",dict(number=trial.number,error=f"{type(exc).__name__}: {exc}"))
        np.savez_compressed(directory/"partial_predictions.npz",gmm=raw)
        raise


def save_trials(study,c,stats):
    records=[dict(number=t.number,state=t.state.name,value=t.value,parameters=t.params,user_attrs=t.user_attrs) for t in study.trials]
    dump(c["output"]/"trials.json",records)
    pd.DataFrame([dict(number=r["number"],state=r["state"],objective_auc=r["value"],**r["parameters"]) for r in records]).to_csv(c["output"]/"trials.csv",index=False)
    stats.update(attempted_trials=len(records),completed_trials=sum(r["state"]=="COMPLETE" for r in records),rejected_trials=sum(r["user_attrs"].get("rejected",False) for r in records),failed_trials=sum(r["state"]=="FAIL" for r in records))
    dump(c["output"]/"run_manifest.json",stats)


def choose(study):
    eligible=[t for t in study.trials if t.state.name=="COMPLETE" and t.value is not None and np.isfinite(t.value)]
    if not eligible:raise RuntimeError("No completed eligible trial")
    return min(eligible,key=lambda t:(-float(t.value),t.number))


def execute(c,optuna_module=None):
    if optuna_module is None:
        import optuna as optuna_module
    out=c["output"];out.mkdir(parents=True,exist_ok=False);d=snapshot.DESIGN;study=None
    stats=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),planned_trials=d["n_trials"],planned_candidate_fits=d["n_trials"]*len(c["groups"])*len(b.DESIGN["initialization_seeds"]),
        attempted_trials=0,completed_trials=0,rejected_trials=0,failed_trials=0,attempted_candidate_fits=0,completed_candidate_fits=0,converged_candidate_fits=0,attempted_projector_fits=0,completed_projector_fits=0,
        competition_test_opened=False,kaggle_upload_performed=False,automatic_promotion=False,fit_accounting="Each candidate fit includes one density GMM initialization and one custom EM. Completed means fit returned; converged counted separately.")
    dump(out/"run_manifest.json",stats)
    try:
        b.check(c["hashes"])
        dump(out/"settings.json",dict(config=c["optuna_config"],fixed_source_design=b.DESIGN,control=CONTROL,input_and_code_sha256=c["hashes"],
            sampler="TPE",pruner="NopPruner",n_jobs=1,objective="Mean10 repetition means of5 hybrid fold AUCs",hybrid_weights=[.5,.5],
            versions={p:version(p) for p in ["numpy","scipy","pandas","scikit-learn","threadpoolctl"]},optuna_version=getattr(optuna_module,"__version__","explicit-test-double")))
        c["frame"].to_csv(out/"cohort.csv",index=False);dump(out/"split_indices.json",c["groups"])
        c["source_reproduction"].to_csv(out/"source_reproduction.csv",index=False)
        study=optuna_module.create_study(direction="maximize",sampler=optuna_module.samplers.TPESampler(seed=d["sampler_seed"],n_startup_trials=d["n_startup_trials"]),pruner=optuna_module.pruners.NopPruner())
        study.enqueue_trial(CONTROL.copy())
        def objective(trial):
            stats["attempted_trials"]=trial.number+1;dump(out/"run_manifest.json",stats)
            b.check(c["hashes"]);params=suggest(trial)
            try:return fit_trial(c,trial,params,stats)
            except (RejectedTrial,np.linalg.LinAlgError,FloatingPointError) as exc:
                if trial.number==0:raise ControlFailure("Control numerical/convergence failure") from exc
                trial.set_user_attr("rejected",True);trial.set_user_attr("reason",f"{type(exc).__name__}: {exc}")
                raise RejectedTrial(str(exc)) from exc
            except ValueError as exc:
                if not numerical_value_error(exc):raise
                if trial.number==0:raise ControlFailure("Control covariance failure") from exc
                trial.set_user_attr("rejected",True);trial.set_user_attr("reason",str(exc));raise RejectedTrial(str(exc)) from exc
        def callback(study,trial):save_trials(study,c,stats)
        study.optimize(objective,n_trials=d["n_trials"],n_jobs=1,catch=(RejectedTrial,),callbacks=[callback])
        save_trials(study,c,stats)
        if len(study.trials)!=d["n_trials"] or any(t.state.name not in ("COMPLETE","FAIL") for t in study.trials):raise RuntimeError("Incomplete30-attempt accounting")
        if not study.trials[0].user_attrs.get("control_reproduced"):raise ControlFailure("Control verification missing")
        best=choose(study);result=b.read(out/"trials"/f"trial_{best.number:04d}"/"result.json")
        primary=next(v for v in result["comparisons"] if v["candidate"]=="candidate_hybrid" and v["reference"]=="original_hybrid")
        is_control=best.number==0;screen=None if is_control else bool(primary["mean_difference"]>snapshot.TOL and primary["wins"]>=6)
        decision=dict(selected_trial=best.number,selected_parameters=best.params,selected_mean_auc=float(best.value),selected_is_control=is_control,
            screening_passed=screen,screening_status="not_applicable_control_selected" if is_control else ("passed" if screen else "failed"),comparisons=result["comparisons"],automatic_promotion=False,
            selection_rule="Highest eligible hybrid mean; exact ties earliest trial; no fallback or extra attempts",
            interpretation="Adaptive development on reused labels/splits; no independent or nested evaluation, no submission or upload")
        dump(out/"best_trial.json",result);dump(out/"decision.json",decision);b.check(c["hashes"])
        stats["status"]="completed"
        stats["artifact_sha256"]={str(p.relative_to(out)):b.digest(p) for p in out.rglob("*") if p.is_file() and p.name!="run_manifest.json"}
        return decision
    except BaseException as exc:
        stats.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
        if study is not None:save_trials(study,c,stats)
        for name in ["decision.json","best_trial.json"]:(out/name).unlink(missing_ok=True)
        raise
    finally:
        stats["finished_utc"]=datetime.now(timezone.utc).isoformat();dump(out/"run_manifest.json",stats)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/generative_optuna_v1.json");p.add_argument("--output");p.add_argument("--dry-run",action="store_true")
    args=p.parse_args();c=snapshot.prepare(args.config,args.output)
    import optuna
    if args.dry_run:print(json.dumps(dict(status="preflight_passed",trials=30,planned_candidate_fits=4500,planned_projector_fits=1500,fits_performed=0,competition_test_opened=False,output_created=False,optuna_version=optuna.__version__,output=str(c["output"])),indent=2))
    else:print(json.dumps(execute(c,optuna),indent=2))

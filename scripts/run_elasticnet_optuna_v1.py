"""Ensemble-oriented Elastic-net Optuna v1; additive and no automatic promotion."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
CONTROL=dict(C=.1,l1_ratio=.75,alpha=.05)
CONTROL_AUC=.9084647196603719
DESIGN=dict(n_trials=30,sampler_seed=42,n_startup_trials=10,C_low=.001,C_high=1.,
            l1_ratio_low=.5,l1_ratio_high=1.,alpha_low=.01,alpha_high=.2)


def en_module():
    from scripts import run_elasticnet_ensemble_v1
    return run_elasticnet_ensemble_v1


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(cfg):
    if set(cfg)!={"suite_config","output_dir","design"} or cfg["design"]!=DESIGN:
        raise ValueError("V1 search design is frozen; changed settings require a separately declared study")
    if any(not isinstance(cfg[k],str) or not cfg[k].strip() for k in ("suite_config","output_dir")):
        raise ValueError("Nonempty paths required")
    return cfg


def prepare(config_path,output=None):
    en=en_module();path=(ROOT/config_path).resolve();cfg=validate(read(path))
    dest=(ROOT/(output or cfg["output_dir"])).resolve()
    if dest.exists():raise FileExistsError("Fresh Optuna output required; no overwrite/resume")
    suite_cfg=(ROOT/cfg["suite_config"]).resolve()
    # Pass the NEW destination: the original completed elasticnet_v1 folder must stay intact.
    c=en.prepare(str(suite_cfg),output=str(dest))
    hashes=dict(c["elasticnet_hashes"])
    hashes.update({str(path):en.digest(path),str(Path(__file__).resolve()):en.digest(__file__)})
    en.check_hashes(hashes)
    c.update(output=dest,optuna_config=cfg,elasticnet_hashes=hashes)
    return c


def suggest(trial):
    return dict(C=trial.suggest_float("C",DESIGN["C_low"],DESIGN["C_high"],log=True),
                l1_ratio=trial.suggest_float("l1_ratio",DESIGN["l1_ratio_low"],DESIGN["l1_ratio_high"]),
                alpha=trial.suggest_float("alpha",DESIGN["alpha_low"],DESIGN["alpha_high"]))


def fit_trial(c,trial,params,manifest,en):
    directory=c["output"]/"trials"/f"trial_{trial.number:04d}"
    directory.mkdir(parents=True,exist_ok=False)
    n=len(c["frame"]);raw=np.empty(n);warnings_saved=[];diagnostics=[]
    en.write(directory/"parameters.json",params)
    with threadpool_limits(limits=1):
        for s in c["groups"]:
            tr,va=s["train_indices"],s["valid_indices"]
            model=en.build_model(params)
            caught=[]
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(c["context"]["x"].iloc[tr],c["context"]["y"].iloc[tr])
            finally:
                warnings_saved.extend(dict(repeat=s["repeat"],fold=s["fold"],category=w.category.__name__,message=str(w.message)) for w in caught)
                en.write(directory/"warnings.json",warnings_saved)
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                raise RuntimeError("ConvergenceWarning; stop study rather than selecting partial trials")
            p=en.probability(model.predict_proba(c["context"]["x"].iloc[va])[:,1],len(va))
            raw[s["start"]:s["stop"]]=p
            classifier=model.named_steps["classifier"]
            diagnostics.append(dict(repeat=s["repeat"],fold=s["fold"],nonzero_coefficients=int(np.count_nonzero(classifier.coef_)),n_iter=int(classifier.n_iter_.max())))
            manifest["completed_fits"]+=1
        blend=(1-params["alpha"])*c["baseline"]+params["alpha"]*raw
    preds=dict(reference=c["baseline"],catboost_frozen=c["catboost_control"],candidate=blend,standalone=raw)
    folds,repeats,summary=en.score(c,preds)
    paired,comparison=en.compare(repeats,["candidate"],"reference")
    paired_cat,comparison_cat=en.compare(repeats,["candidate"],"catboost_frozen")
    scores=summary.set_index("candidate")
    value=float(scores.loc["candidate","mean_auc"])
    if trial.number==0:
        if params!=CONTROL:raise ValueError("Trial0 must be the queued control")
        if not np.isclose(value,CONTROL_AUC,rtol=0,atol=en.TOL):
            raise ValueError(f"Control failed reproduction: actual={value:.17g}, expected={CONTROL_AUC:.17g}")
    stats=comparison.iloc[0].to_dict();cat_stats=comparison_cat.iloc[0].to_dict()
    result=dict(number=trial.number,state="COMPLETE",parameters=params,objective_auc=value,
                standalone_auc=float(scores.loc["standalone","mean_auc"]),repeat_sd=float(scores.loc["candidate","repeat_sd"]),
                reference_comparison=stats,catboost_comparison=cat_stats,
                control_reproduced=True if trial.number==0 else None)
    for name,df in [("fold_scores",folds),("repeat_scores",repeats),("model_summary",summary),("comparison",comparison),
                    ("comparison_vs_catboost",comparison_cat),("paired_repeat_differences",paired),("paired_vs_catboost",paired_cat)]:
        df.to_csv(directory/(name+".csv"),index=False)
    en.write(directory/"diagnostics.json",diagnostics);en.write(directory/"result.json",result)
    np.savez_compressed(directory/"predictions.npz",elasticnet=raw,ensemble=blend)
    en.check_hashes(c["elasticnet_hashes"])
    trial.set_user_attr("objective_auc",value)
    trial.set_user_attr("reference_comparison",en.safe(stats))
    trial.set_user_attr("catboost_comparison",en.safe(cat_stats))
    if trial.number==0:trial.set_user_attr("control_reproduced",True)
    print(f"trial={trial.number} ensemble_auc={value:.9f} fits={manifest['completed_fits']}/{manifest['planned_fits']}",flush=True)
    return value


def records(study):
    return [dict(number=t.number,state=t.state.name,value=t.value,parameters=t.params,user_attrs=t.user_attrs) for t in study.trials]


def select_best(trials):
    complete=[t for t in trials if t.state.name=="COMPLETE" and t.value is not None and np.isfinite(t.value)]
    if len(complete)!=DESIGN["n_trials"]:raise ValueError("All30 trials must complete before final selection")
    return min(complete,key=lambda t:(-float(t.value),t.number))


def save_trials(study,out,en):
    rows=records(study);en.write(out/"trials.json",rows)
    pd.DataFrame([dict(number=r["number"],state=r["state"],objective_auc=r["value"],**r["parameters"]) for r in rows]).to_csv(out/"trials.csv",index=False)


def execute(c,optuna_module=None):
    en=en_module()
    if optuna_module is None:
        import optuna as optuna_module
    out=c["output"];out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status="running",planned_trials=DESIGN["n_trials"],planned_fits=DESIGN["n_trials"]*len(c["groups"]),completed_fits=0,
                  test_opened=False,kaggle_upload_performed=False,started_utc=datetime.now(timezone.utc).isoformat())
    study=None
    en.write(out/"run_manifest.json",manifest)
    try:
        en.write(out/"settings.json",dict(config=c["optuna_config"],input_and_code_sha256=c["elasticnet_hashes"],
              objective="Mean ensemble fold AUC over ten repetitions; addition to original ensemble ONLY",
              control=CONTROL,control_auc=CONTROL_AUC,solver="saga",max_iter=20000,tol=1e-4,model_seed=42,
              sampler="TPE",pruner="NopPruner",n_jobs=1,versions={"optuna":getattr(optuna_module,"__version__","test-double"),"scikit-learn":version("scikit-learn")}))
        c["frame"].to_csv(out/"cohort.csv",index=False);en.write(out/"split_indices.json",c["groups"])
        study=optuna_module.create_study(direction="maximize",sampler=optuna_module.samplers.TPESampler(seed=DESIGN["sampler_seed"],n_startup_trials=DESIGN["n_startup_trials"]),pruner=optuna_module.pruners.NopPruner())
        study.enqueue_trial(CONTROL.copy())
        def objective(trial):
            en.check_hashes(c["elasticnet_hashes"])
            params=suggest(trial)
            value=fit_trial(c,trial,params,manifest,en)
            en.write(out/"run_manifest.json",manifest)
            return value
        def callback(study,trial):save_trials(study,out,en)
        # Exceptions stop the study. No catch-and-continue or pruning of partial trials.
        study.optimize(objective,n_trials=DESIGN["n_trials"],n_jobs=1,callbacks=[callback])
        best=select_best(study.trials)
        if study.trials[0].user_attrs.get("control_reproduced") is not True:
            raise ValueError("Control reproduction did not pass")
        result=read(out/"trials"/f"trial_{best.number:04d}"/"result.json")
        stat=result["reference_comparison"];catstat=result["catboost_comparison"]
        passed=lambda x:bool(x["mean_difference"]>en.TOL and x["wins"]>=6)
        decision=dict(selected_trial=best.number,selected_mean_auc=float(best.value),parameters=best.params,
                      screening_passed=passed(stat),screening_both_controls=bool(passed(stat) and passed(catstat)),
                      reference_comparison=stat,catboost_control_comparison=catstat,control_reproduced=True,
                      automatic_promotion=False,selection_rule="Highest ensemble mean; exact ties earliest trial; no fallback",
                      interpretation="Adaptive TPE development tuning, not nested CV or independent confirmation. No pruning, no test data, no submission.")
        en.write(out/"best_trial.json",result);en.write(out/"decision.json",decision)
        save_trials(study,out,en)
        en.check_hashes(c["elasticnet_hashes"])
        manifest["status"]="completed"
        manifest["completed_trials"]=len([t for t in study.trials if t.state.name=="COMPLETE"])
        manifest["artifact_sha256"]={str(p.relative_to(out)):en.digest(p) for p in out.rglob("*") if p.is_file() and p!=out/"run_manifest.json"}
        return dict(status="completed",output=str(out),completed_fits=manifest["completed_fits"],automatic_promotion=False)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
        if study is not None:save_trials(study,out,en)
        raise
    finally:
        manifest["finished_utc"]=datetime.now(timezone.utc).isoformat();en.write(out/"run_manifest.json",manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/elasticnet_optuna_v1.json")
    p.add_argument("--output")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args();c=prepare(args.config,args.output)
    import optuna
    if args.dry_run:
        print(json.dumps(dict(status="preflight_passed",trials=30,planned_fits=1500,fits_performed=0,
                             first_trial=CONTROL,first_trial_expected_auc=CONTROL_AUC,optuna_version=optuna.__version__,test_opened=False,output=str(c["output"])),indent=2))
    else:print(json.dumps(execute(c,optuna),indent=2))


if __name__=="__main__":main()

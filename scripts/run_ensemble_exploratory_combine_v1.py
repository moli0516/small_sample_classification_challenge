"""Post-hoc exploratory combinations. Do not change the original combination gate."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scripts import run_ensemble_research_suite_v1 as suite

ROOT=Path(__file__).resolve().parents[1]


def compose(base,cat,alpha,lgb,xgb,stack):
    base=np.asarray(base,dtype=float)
    if base.ndim!=2 or base.shape[1]!=3: raise ValueError("Expected three baseline components")
    n=len(base)
    for p in [base[:,i] for i in range(3)]+[cat,lgb,xgb,stack]:suite.probability(p,n)
    if alpha!=.1: raise ValueError("This experiment fixes the selected CatBoost alpha to 0.10")
    def blend(matrix):return (1-alpha)*matrix.mean(axis=1)+alpha*cat
    original_cat=blend(base)
    first=base.copy();first[:,2]=lgb
    second=base.copy();second[:,1]=xgb
    both=base.copy();both[:,1]=xgb;both[:,2]=lgb
    candidates={"cat_plus_lgbm_es":blend(first),"cat_plus_xgb_es":blend(second),
                "cat_plus_both_es":blend(both),"cat_plus_stacking":(1-alpha)*stack+alpha*cat}
    return original_cat,candidates


def prepare(config_path,output=None):
    c=suite.prepare(config_path)
    dest=(ROOT/(output or str(c["output_root"]/"exploratory_combine_v1"))).resolve()
    if dest.exists(): raise FileExistsError("Fresh exploratory output required")
    if any(dest.is_relative_to(c["output_root"]/task) or (c["output_root"]/task).is_relative_to(dest)
           for task in ("catboost","lgbm_es","xgb_es","stacking")):
        raise ValueError("Output must not overlap source study trees")
    children={};hashes=dict(c["hashes"])
    for task in ("catboost","lgbm_es","xgb_es","stacking"):
        d,z,h=suite.load_child(c,task);children[task]=(d,z);hashes.update(h)
    cat_dec,cat_z=children["catboost"];spec=cat_dec["selected_spec"]
    expected=dict(depth=4,iterations=100,learning_rate=.1,l2_leaf_reg=3)
    if cat_dec["selected_candidate"]!="cat_10_alpha0.1" or spec["alpha"]!=.1 or any(spec["cat_parameters"][k]!=v for k,v in expected.items()):
        raise ValueError("Expected frozen selected CatBoost candidate; no re-selection")
    cat=cat_z["component__"+spec["component_key"]]
    lgb=children["lgbm_es"][1]["component__lightgbm"]
    xgb=children["xgb_es"][1]["component__xgboost"]
    sd,sz=children["stacking"];stack=sz["prediction__"+sd["selected_candidate"]]
    control,preds=compose(c["base"],cat,.1,lgb,xgb,stack)
    saved=cat_z["prediction__"+cat_dec["selected_candidate"]]
    if not np.allclose(control,saved,rtol=0,atol=suite.TOL): raise ValueError("Frozen CatBoost control probabilities differ")
    _,_,summary,_,_=suite.score(c,{"reference":c["baseline"],"catboost_frozen":control})
    mean=float(summary.set_index("candidate").loc["catboost_frozen","mean_auc"])
    if not np.isclose(mean,cat_dec["selected_mean_auc"],rtol=0,atol=suite.TOL):
        raise ValueError("Frozen CatBoost control AUC does not reproduce")
    hashes[str(Path(__file__).resolve())]=suite.digest(__file__)
    suite.check_hashes(hashes)
    return c,dest,preds,control,hashes,children


def run(config_path,output=None,dry_run=False):
    c,dest,preds,control,hashes,children=prepare(config_path,output)
    if dry_run:return dict(status="preflight_passed",candidates=list(preds),fits_performed=0,test_opened=False,output=str(dest))
    dest.mkdir(parents=True,exist_ok=False)
    manifest=dict(status="running",fits_performed=0,test_opened=False,kaggle_upload_performed=False,
                  exploratory_post_hoc=True,started_utc=datetime.now(timezone.utc).isoformat())
    suite.write(dest/"run_manifest.json",manifest)
    try:
        specs={name:dict(cat_alpha=.1,original_component_weights="equal thirds before CatBoost blend") for name in preds}
        all_preds={"reference":c["baseline"],"catboost_frozen":control,**preds}
        folds,repeats,summary,paired,comparison=suite.score(c,all_preds)
        cc=dict(c,baseline=control)
        _,_,_,cat_paired,cat_comparison=suite.score(cc,{"reference":control,**preds})
        cat_comparison["reference"]="catboost_frozen"
        decision=suite.decision_for(summary,comparison,specs)
        selected=decision["selected_candidate"]
        extra=cat_comparison[cat_comparison.candidate==selected].iloc[0].to_dict()
        decision.update(exploratory_post_hoc=True,catboost_control_comparison=extra,
                        beats_catboost_control_screening=bool(extra["mean_difference"]>suite.TOL and extra["wins"]>=6),
                        exploratory_screening_both_controls=bool(decision["screening_passed"] and extra["mean_difference"]>suite.TOL and extra["wins"]>=6),
                        automatic_promotion=False,
                        interpretation="Post-hoc combinations include individually failed changes. Original suite screening/combination rules are unchanged. Compare both controls; no independent confirmation or significance claim.")
        for name,df in (("fold_scores",folds),("repeat_scores",repeats),("model_summary",summary),
                        ("comparison_vs_original",comparison),("paired_vs_original",paired),
                        ("comparison_vs_catboost",cat_comparison),("paired_vs_catboost",cat_paired)):
            df.to_csv(dest/(name+".csv"),index=False)
        c["frame"].to_csv(dest/"cohort.csv",index=False)
        np.savez_compressed(dest/"predictions.npz",**all_preds)
        suite.write(dest/"settings.json",dict(config_path=str(config_path),input_and_code_sha256=hashes,
                    child_decisions={task:d for task,(d,_) in children.items()},candidate_specs=specs,
                    selection="Highest mean of four exploratory combinations; exact ties use listed order; no fallback"))
        suite.write(dest/"decision.json",decision)
        suite.check_hashes(hashes)
        manifest["status"]="completed"
        manifest["artifact_sha256"]={p.name:suite.digest(p) for p in dest.iterdir() if p.is_file() and p.name!="run_manifest.json"}
        return dict(status="completed",fits_performed=0,output=str(dest),automatic_promotion=False)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_utc"]=datetime.now(timezone.utc).isoformat()
        suite.write(dest/"run_manifest.json",manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/ensemble_research_suite_v1.json")
    parser.add_argument("--output")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    print(__import__("json").dumps(run(args.config,args.output,args.dry_run),indent=2))


if __name__=="__main__":main()

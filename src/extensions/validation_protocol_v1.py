"""Fixed-procedure validation sensitivity study. No tuning or competition test use."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import inspect
import json
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

TOL = 1e-12
MODELS = ["l1_c01", "l1_c1", "l1_xgb", "l1_xgb_lgbm"]
SCHEMES = ["holdout20", "cv3", "cv5", "cv10"]
PAIRS = [("ensemble", "l1_xgb", "l1_xgb_lgbm"), ("linear", "l1_c01", "l1_c1")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(cfg):
    fields = {"source_study", "sparse_l1_config", "output_dir", "split_seeds", "holdout_fraction", "k_values"}
    if not isinstance(cfg, dict) or set(cfg) != fields:
        raise ValueError(f"Expected exactly {sorted(fields)}")
    for name in ("source_study", "sparse_l1_config", "output_dir"):
        if not isinstance(cfg[name], str) or not cfg[name].strip():
            raise ValueError("Nonempty paths required.")
    if (cfg["split_seeds"] != list(range(10)) or any(type(s) is not int for s in cfg["split_seeds"])
        or cfg["k_values"] != [3,5,10] or any(type(k) is not int for k in cfg["k_values"])
        or type(cfg["holdout_fraction"]) not in (int,float) or cfg["holdout_fraction"] != .2):
        raise ValueError("V1 fixes seeds0..9, k3/5/10 and holdout20%; changed design requires another version.")
    return cfg


def verify_artifacts(root, helper, required):
    root = root.resolve()
    path = root / "run_manifest.json"
    manifest = read(path)
    checks = manifest.get("artifact_sha256", {})
    if manifest.get("status") != "completed" or not required <= checks.keys():
        raise ValueError("Source study incomplete or required checksums missing.")
    paths = [path]
    for name, digest in checks.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or helper._sha(p) != digest:
            raise ValueError(f"Source artifact changed/unsafe: {name}")
        paths.append(p)
    return manifest, paths


def make_splits(y, cfg):
    """Generate once, share the same training/validation indices across all models."""
    records = []
    dummy = np.zeros((len(y),1))
    for scheme in SCHEMES:
        for repeat, seed in enumerate(cfg["split_seeds"]):
            splitter = (StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
                if scheme == "holdout20" else StratifiedKFold(n_splits=int(scheme[2:]), shuffle=True, random_state=seed))
            for fold, (tr, va) in enumerate(splitter.split(dummy,y)):
                records.append(dict(scheme=scheme, repeat=repeat, split_seed=seed, fold=fold,
                                    train_indices=tr.tolist(), valid_indices=va.tolist()))
    validate_splits(records,y)
    return records


def validate_splits(splits, y):
    if len(splits) != 190:
        raise ValueError("Expected190 train/validation assignments.")
    keys = [(s["scheme"],s["repeat"],s["fold"]) for s in splits]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate split keys.")
    all_indices = set(range(len(y)))
    for scheme in SCHEMES:
        k = 1 if scheme == "holdout20" else int(scheme[2:])
        for r in range(10):
            group = [s for s in splits if s["scheme"] == scheme and s["repeat"] == r]
            if len(group) != k or {s["fold"] for s in group} != set(range(k)):
                raise ValueError("Incomplete split group.")
            validation_rows = []
            for s in group:
                tr, va = s["train_indices"],s["valid_indices"]
                if (s["split_seed"] != r or any(type(i) is not int for i in tr+va)
                    or len(tr) != len(set(tr)) or len(va) != len(set(va))
                    or set(tr) & set(va) or set(tr) | set(va) != all_indices):
                    raise ValueError("Invalid split partition/seed.")
                if len(np.unique(np.asarray(y)[tr])) != 2 or len(np.unique(np.asarray(y)[va])) != 2:
                    raise ValueError("Each partition needs both classes for AUC.")
                validation_rows.extend(va)
            if k > 1 and sorted(validation_rows) != list(range(len(y))):
                raise ValueError("Each CV repeat must validate every row exactly once.")


def summarize(folds):
    keys = ["scheme","repeat","split_seed","fold","candidate"]
    if not set(keys+["auc","train_rows","valid_rows"]) <= set(folds) or folds.duplicated(keys).any():
        raise ValueError("Invalid fold score table.")
    if len(folds) != 760 or set(folds.candidate) != set(MODELS) or set(folds.scheme) != set(SCHEMES):
        raise ValueError("Need all four procedures on190 splits; partial scores are not accepted.")
    values = folds.auc.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid AUC values.")
    for scheme in SCHEMES:
        k = 1 if scheme == "holdout20" else int(scheme[2:])
        expected = {(r,r,f) for r in range(10) for f in range(k)}
        for name in MODELS:
            part = folds.loc[(folds.scheme == scheme) & (folds.candidate == name)]
            if set(part[["repeat","split_seed","fold"]].itertuples(index=False,name=None)) != expected:
                raise ValueError("Missing or unpaired validation scores.")
    size_variation = folds.groupby(["scheme","repeat","split_seed","fold"])[["train_rows","valid_rows"]].nunique()
    if (size_variation > 1).any().any():
        raise ValueError("Candidates were evaluated on different partition sizes.")
    repeats = folds.groupby(["scheme","repeat","split_seed","candidate"],as_index=False).auc.mean()
    repeats = repeats.rename(columns={"auc":"mean_fold_auc"})
    summary_rows, paired_rows, pair_summaries = [], [], []
    for scheme in SCHEMES:
        for name in MODELS:
            part = repeats.loc[(repeats.scheme==scheme)&(repeats.candidate==name)]
            raw = folds.loc[(folds.scheme==scheme)&(folds.candidate==name)]
            scores = part.mean_fold_auc
            summary_rows.append(dict(scheme=scheme,candidate=name,repeats=10,
                mean_auc=float(scores.mean()),repeat_sd=float(scores.std(ddof=1)),
                minimum_repeat_auc=float(scores.min()),maximum_repeat_auc=float(scores.max()),
                median_repeat_auc=float(scores.median()),train_rows_min=int(raw.train_rows.min()),
                train_rows_max=int(raw.train_rows.max()),valid_rows_min=int(raw.valid_rows.min()),valid_rows_max=int(raw.valid_rows.max())))
        for pair, a, b in PAIRS:
            def get(name):
                return repeats.loc[(repeats.scheme==scheme)&(repeats.candidate==name)].set_index(["repeat","split_seed"]).mean_fold_auc.sort_index()
            first, second = get(a),get(b)
            delta = second-first
            for (r,seed), d in delta.items():
                paired_rows.append(dict(scheme=scheme,pair=pair,repeat=int(r),split_seed=int(seed),
                    model_a=a,model_b=b,auc_a=float(first.loc[(r,seed)]),auc_b=float(second.loc[(r,seed)]),difference=float(d)))
            wins,losses,ties = int((delta>TOL).sum()),int((delta< -TOL).sum()),int((abs(delta)<=TOL).sum())
            pair_summaries.append(dict(scheme=scheme,pair=pair,model_a=a,model_b=b,
                mean_difference=float(delta.mean()),difference_sd=float(delta.std(ddof=1)),
                minimum_difference=float(delta.min()),maximum_difference=float(delta.max()),
                wins_b=wins,wins_a=losses,ties=ties,winner_changes_across_splits=bool(wins>0 and losses>0),
                b_win_fraction=wins/10,sd_over_001=float(delta.std(ddof=1)/.01)))
    return repeats,pd.DataFrame(summary_rows),pd.DataFrame(paired_rows),pd.DataFrame(pair_summaries)


def plot_results(repeats, paired, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,(pair,a,b) in zip(axes,PAIRS):
        arrays=[];positions=[];colors=[]
        for j,scheme in enumerate(SCHEMES):
            for offset,name,color in ((-.17,a,"#3377aa"),(.17,b,"#dd8844")):
                data=repeats.loc[(repeats.scheme==scheme)&(repeats.candidate==name),"mean_fold_auc"].to_numpy()
                arrays.append(data);positions.append(j+offset);colors.append(color)
                ax.scatter(j+offset+np.linspace(-.04,.04,len(data)),data,color=color,s=13,zorder=3)
        boxes=ax.boxplot(arrays,positions=positions,widths=.26,patch_artist=True,showfliers=False)
        for patch,color in zip(boxes["boxes"],colors):patch.set_facecolor(color);patch.set_alpha(.3)
        ax.set_xticks(range(4),["Holdout20%","3-fold","5-fold","10-fold"])
        ax.set_title(f"{pair}: {a} (blue) vs {b} (orange)",fontsize=9)
        ax.set_ylabel("Repeat-level mean fold AUC");ax.grid(axis="y",alpha=.2)
    fig.tight_layout();path=out/"auc_spread.png";fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,(pair,a,b) in zip(axes,PAIRS):
        arrays=[paired.loc[(paired.scheme==scheme)&(paired.pair==pair),"difference"].to_numpy() for scheme in SCHEMES]
        ax.boxplot(arrays,positions=range(4),widths=.45,showfliers=False)
        for j,data in enumerate(arrays):ax.scatter(j+np.linspace(-.07,.07,len(data)),data,s=18)
        ax.axhline(0,color="black",linewidth=1)
        ax.axhline(.01,color="grey",linestyle=":",linewidth=.8)
        ax.axhline(-.01,color="grey",linestyle=":",linewidth=.8)
        ax.set_xticks(range(4),["Holdout20%","3-fold","5-fold","10-fold"])
        ax.set_title(f"{pair}: {b} minus {a}",fontsize=9)
        ax.set_ylabel("Paired repeat AUC difference");ax.grid(axis="y",alpha=.2)
    fig.tight_layout();path=out/"paired_differences.png";fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    return figures


def full_check(c):
    for path,digest in c["hashes"].items():
        if c["helper"]._sha(path)!=digest:raise RuntimeError(f"Input/code/artifact changed: {path}")


def fingerprint(paths):
    return {str(p):(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}


class ValidationProtocolV1:
    def _prepare(self,config_path,output=None):
        from src.experiments.lgbm_optuna_experiment import LGBMOptunaExperiment
        from src.submission.frozen_xgboost_submission import FrozenXGBoostSubmission
        from src.config.experiment_config import ExperimentConfig
        from src.modeling.pipeline_factory import PipelineFactory
        from src.modeling import registered_models
        from src.modeling.model_registry import ModelRegistry
        h=FrozenXGBoostSubmission();path=h._path(config_path)
        cfg=validate(read(path));out=h._path(output or cfg["output_dir"])
        if out.exists():raise FileExistsError("New output directory required; no overwrite/resume.")
        root=h._path(cfg["source_study"])
        manifest,paths=verify_artifacts(root,h,{"settings.json","decision.json","best_trial.json"})
        settings,best,decision=(read(root/n) for n in ("settings.json","best_trial.json","decision.json"))
        if (manifest.get("control_trial_reproduced") is not True or decision.get("selected_trial")!=14
            or decision.get("promote") is not True or best.get("number")!=14 or best.get("state")!="COMPLETE"
            or best.get("reference_reproduced") is not True or not np.isclose(best["objective_auc"],.9083364180103309,rtol=0,atol=TOL)):
            raise ValueError("Need the original promoted Trial14 source.")
        original=LGBMOptunaExperiment()._prepare(settings["config_path"],str(out))
        if (original["hashes"]!=settings["input_and_code_sha256"] or original["versions"]!=settings["versions"]
            or original["config"]!=settings["config"]):
            raise ValueError("Historical inputs/config/code/environment changed; do not bypass provenance.")
        ctx=original["context"]
        if len(ctx["x"])!=250 or ctx["x"].shape[1]!=150 or ctx["candidate"]["k"]!=20:
            raise ValueError("Expected original250-row/150-feature/top20 problem.")
        trial=h._path(best["cv_run"])
        if trial!=(root/"trials/trial_0014/cv").resolve() or h._sha(trial/"run_manifest.json")!=best["registry_manifest_sha256"]:
            raise ValueError("Selected trial path/manifest mismatch.")
        tm,tp=verify_artifacts(trial,h,{"settings.json","fold_scores.csv","comparison.csv"})
        if tm.get("completed_folds")!=50 or tm.get("control_auc_reproduced") is not True:
            raise ValueError("Trial14 is not a complete verified50-fold source.")
        resolved=read(trial/"settings.json")["config"]
        expected=json.loads(json.dumps(original["base"]))
        expected["models"]["lightgbm"]["parameters"]=best["parameters"];expected["output_dir"]=str(trial)
        if resolved!=expected:raise ValueError("Selected resolved procedure differs from frozen source.")
        sparse_path=h._path(cfg["sparse_l1_config"]);sparse=ExperimentConfig.from_json(sparse_path)
        dense=ctx["linear"]
        if (sparse.regularization_c!=.1 or dense.regularization_c!=1. or sparse.regularization_type!="l1"
            or sparse.feature_selection_k is not None or sparse.model_seed!=42 or dense.model_seed!=42):
            raise ValueError("Expected original all-feature L1 C.1/C1 with model seed42.")
        ignored={"experiment_id","regularization_c","output_dir","n_splits","split_seeds"}
        if {k:v for k,v in sparse.to_dict().items() if k not in ignored}!={k:v for k,v in dense.to_dict().items() if k not in ignored}:
            raise ValueError("L1 model procedures differ beyond C; inspect rather than relabel.")
        xgb=ctx["candidate"]["estimator_parameters"];lgb=best["parameters"]
        if xgb.get("random_state")!=42 or lgb.get("random_state")!=42:
            raise ValueError("Keep established estimator seeds42 fixed.")
        splits=make_splits(ctx["y"],cfg)
        # Reproduction applies only when BOTH ordered train/validation indices match.
        old={(s["repeat"],s["fold"]):s for s in ctx["splits"]}
        ref=pd.read_csv(trial/"fold_scores.csv",float_precision="round_trip")
        if len(ref)!=50 or ref.duplicated(["repeat","fold"]).any():raise ValueError("Invalid source fold keys.")
        ref=ref.set_index(["repeat","fold"])
        if not ref.index.is_unique or set(ref.index)!=set(old):raise ValueError("Source fold keys changed.")
        matches={}
        for s in splits:
            if s["scheme"]!="cv5":continue
            o=old[(s["repeat"],s["fold"])]
            matches[(s["repeat"],s["fold"])]=bool(s["split_seed"]==o["split_seed"] and s["train_indices"]==o["train_indices"] and s["valid_indices"]==o["valid_indices"])
        versions=dict(original["versions"],matplotlib=version("matplotlib"))
        paths+=tp+[path,sparse_path,Path(__file__),h._path("scripts/run_validation_protocol_v1.py"),
            Path(inspect.getfile(ModelRegistry)),Path(inspect.getfile(registered_models)),Path(inspect.getfile(PipelineFactory))]
        hashes=dict(original["hashes"]);hashes.update({str(p.resolve()):h._sha(p) for p in paths})
        return dict(helper=h,context=ctx,config=cfg,output=out,sparse=sparse,splits=splits,
            lightgbm_parameters=lgb,hashes=hashes,versions=versions,reference=ref,reference_matches=matches)

    def _models(self,c):
        from src.modeling.pipeline_factory import PipelineFactory
        from src.modeling.registered_models import L1Model,XGBTopKModel,LightGBMModel
        return [("l1_c01",PipelineFactory().create(c["sparse"]),1),
            ("l1_c1",L1Model(context=c["context"]),1),("xgboost",XGBTopKModel(context=c["context"]),2),
            ("lightgbm",LightGBMModel(parameters=c["lightgbm_parameters"]),1)]

    def run(self,config_path,output=None,dry_run=False):
        c=self._prepare(config_path,output);full_check(c);stamps=fingerprint(c["hashes"])
        if dry_run:return dict(status="preflight_passed",schemes=SCHEMES,repeats_per_scheme=10,
            validation_splits=190,planned_estimator_fits=950,scored_procedures=MODELS,
            matching_historical_cv5_splits=sum(c["reference_matches"].values()),
            all_splits_refitted=True,test_opened=False,output=str(c["output"]))
        h,out,ctx=c["helper"],c["output"],c["context"]
        out.mkdir(parents=True,exist_ok=False)
        manifest=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),planned_splits=190,
            completed_splits=0,planned_estimator_fits=950,attempted_estimator_fits=0,completed_estimator_fits=0,
            test_opened=False,kaggle_upload_performed=False)
        rows,predictions,diagnostics,warnrows,reproduction=[],[],[],[],[]
        started=time.monotonic()
        h._write(out/"run_manifest.json",manifest)
        try:
            h._write(out/"settings.json",dict(config=c["config"],versions=c["versions"],input_and_code_sha256=c["hashes"],
                sparse_l1=c["sparse"].to_dict(),dense_l1=ctx["linear"].to_dict(),
                xgboost_parameters=ctx["candidate"]["estimator_parameters"],lightgbm_parameters=c["lightgbm_parameters"],
                xgboost_k=20,ensemble_weights=dict(l1_xgb=[1,1],l1_xgb_lgbm=[1,1,1]),
                interpretation="Fixed selected procedures; every split refitted. Adaptive sensitivity study, not nested/independent final evaluation."))
            h._write(out/"split_indices.json",c["splits"]);h._write(out/"warnings.json",[])
            for s in c["splits"]:
                if fingerprint(stamps)!=stamps:raise RuntimeError("Source file metadata changed during study.")
                tr,va=s["train_indices"],s["valid_indices"]
                key={n:s[n] for n in ("scheme","repeat","split_seed","fold")}
                probabilities={}
                for alias,model,cost in self._models(c):
                    caught=[];manifest["attempted_estimator_fits"]+=cost
                    h._write(out/"run_manifest.json",manifest)
                    try:
                        with warnings.catch_warnings(record=True) as caught,(out/"fit.log").open("a",encoding="utf-8") as log:
                            warnings.simplefilter("always")
                            with redirect_stdout(log),redirect_stderr(log):
                                model.fit(ctx["x"].iloc[tr],ctx["y"].iloc[tr])
                                p=(model.positive_probability(ctx["x"].iloc[va]) if hasattr(model,"positive_probability")
                                    else h._probabilities(model,ctx["x"].iloc[va]))
                    finally:
                        warnrows.extend(dict(key,model=alias,category=w.category.__name__,message=str(w.message)) for w in caught)
                        h._write(out/"warnings.json",warnrows)
                    if caught:raise RuntimeError("Python warnings require review; partial scores are not accepted.")
                    p=np.asarray(p,dtype=float)
                    if p.shape!=(len(va),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("Invalid component probability.")
                    probabilities[alias]=p;manifest["completed_estimator_fits"]+=cost
                    detail=model.diagnostics() if hasattr(model,"diagnostics") else {"feature_count":len(ctx["x"].columns)}
                    diagnostics.append(dict(key,model=alias,**detail))
                probabilities["l1_xgb"]=(probabilities["l1_c1"]+probabilities["xgboost"])/2
                probabilities["l1_xgb_lgbm"]=(probabilities["l1_c1"]+probabilities["xgboost"]+probabilities["lightgbm"])/3
                auc={name:float(roc_auc_score(ctx["y"].iloc[va],p)) for name,p in probabilities.items()}
                if s["scheme"]=="cv5":
                    matched=c["reference_matches"][(s["repeat"],s["fold"])]
                    for new,old in (("l1_c1","l1"),("xgboost","xgboost"),("lightgbm","lightgbm"),("l1_xgb","l1_xgb"),("l1_xgb_lgbm","l1_xgb_lgbm")):
                        expected=float(c["reference"].loc[(s["repeat"],s["fold"]),old])
                        ok=bool(np.isclose(auc[new],expected,rtol=0,atol=TOL)) if matched else None
                        reproduction.append(dict(key,candidate=new,ordered_split_matches=matched,source_auc=expected,current_auc=auc[new],matches=ok))
                        if matched and not ok:raise RuntimeError("Identical cv5 split failed historical AUC reproduction; do not relax tolerance.")
                rows.extend(dict(key,candidate=name,auc=auc[name],train_rows=len(tr),valid_rows=len(va)) for name in MODELS)
                for j,i in enumerate(va):
                    predictions.append(dict(key,row_index=int(i),id=str(ctx["ids"].iloc[i]),target=int(ctx["y"].iloc[i]),
                        **{name:float(p[j]) for name,p in probabilities.items()}))
                manifest["completed_splits"]+=1
                manifest["elapsed_seconds"]=round(time.monotonic()-started,3)
                h._write(out/"run_manifest.json",manifest)
                print(f"[{manifest['completed_splits']}/190] {s['scheme']} repeat={s['repeat']} fold={s['fold']} fits={manifest['completed_estimator_fits']}/950 elapsed={manifest['elapsed_seconds']:.1f}s",flush=True)
                if (s["scheme"]=="holdout20" or s["fold"]==int(s["scheme"][2:])-1):
                    pd.DataFrame(rows).to_csv(out/"fold_scores.csv",index=False)
                    pd.DataFrame(predictions).to_csv(out/"validation_probabilities.csv",index=False)
                    h._write(out/"diagnostics.json",diagnostics)
            full_check(c)
            repeats,table,paired,comparison=summarize(pd.DataFrame(rows))
            repeats.to_csv(out/"repeat_scores.csv",index=False);table.to_csv(out/"model_summary.csv",index=False)
            paired.to_csv(out/"paired_repeat_differences.csv",index=False);comparison.to_csv(out/"paired_summary.csv",index=False)
            pd.DataFrame(reproduction).to_csv(out/"cv5_reproduction.csv",index=False)
            plot_results(repeats,paired,out)
            h._write(out/"interpretation.json",dict(status="ready_for_analysis",automatic_protocol_selection=False,
                question="Which split scheme provides stable evidence for a ~0.01 AUC difference and the actual ensemble decision?",
                cautions=["Different k changes training size.","Repeats overlap; SD is not a confidence interval.",
                    "Win fractions are not probabilities of being truly best.","Previously tuned fixed candidates; no independent/nested evaluation.",
                    "Public leaderboard is not ground truth; no test data read."]))
            full_check(c)
            manifest.update(status="completed",historical_cv5_matching_splits=sum(c["reference_matches"].values()),
                artifact_sha256={p.name:h._sha(p) for p in out.iterdir() if p.is_file() and p.name!="run_manifest.json"})
            return dict(status="completed",output=str(out),validation_splits=190,estimator_fits=950,
                        automatic_protocol_selection=False,test_opened=False)
        except BaseException as exc:
            manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
            pd.DataFrame(rows).to_csv(out/"partial_fold_scores.csv",index=False)
            pd.DataFrame(predictions).to_csv(out/"partial_validation_probabilities.csv",index=False)
            pd.DataFrame(reproduction).to_csv(out/"cv5_reproduction.csv",index=False)
            h._write(out/"diagnostics.json",diagnostics)
            h._write(out/"error.json",dict(error=manifest["error"],traceback=traceback.format_exc()))
            raise
        finally:
            manifest["finished_utc"]=datetime.now(timezone.utc).isoformat();h._write(out/"run_manifest.json",manifest)

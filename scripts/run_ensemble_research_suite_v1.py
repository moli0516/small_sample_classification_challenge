"""Ensemble research suite v1. Frozen reference; no submissions or auto-promotion."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12
REFERENCE_AUC = .9083364180103309
TASKS = ("catboost", "weights", "lgbm_es", "xgb_es", "combine", "stacking")
NAMES = ("l1", "xgboost", "lightgbm")
WEIGHTS = [(1/3, 1/3, 1/3), (.5, .25, .25), (.25, .5, .25), (.25, .25, .5)]
ALPHAS = [.05, .1, .25]
DESIGN = dict(version=1, cat_depth=[2,4], cat_iterations=[100,300], cat_learning_rate=[.03,.1],
              cat_l2=[3,10], alphas=ALPHAS, weights=[list(w) for w in WEIGHTS],
              inner_fraction=.2, inner_seed_base=10000, early_stop_cap=1000, early_stop_patience=30,
              stacking_inner_folds=3, stacking_seed_base=40000, stacking_meta_C=.1)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [safe(v) for v in x]
    if isinstance(x, np.generic): x=x.item()
    if isinstance(x, float) and not np.isfinite(x): return str(x)
    return x


def write(p, x):
    Path(p).write_text(json.dumps(safe(x), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def versions():
    result={}
    for p in ("numpy","pandas","scikit-learn","xgboost","lightgbm","catboost","threadpoolctl"):
        try: result[p]=version(p)
        except PackageNotFoundError: result[p]=None
    return result


def check_hashes(hashes):
    for p,h in hashes.items():
        if digest(p)!=h: raise ValueError("Source/code/input changed: "+p)


def verify_artifacts(root, required):
    root=Path(root).resolve(); manifest=read(root/"run_manifest.json")
    hashes=manifest.get("artifact_sha256",{})
    if manifest.get("status")!="completed" or not required<=hashes.keys():
        raise ValueError("Completed source with required hashes needed: "+str(root))
    result={str(root/"run_manifest.json"):digest(root/"run_manifest.json")}
    for name,h in hashes.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or digest(p)!=h:
            raise ValueError("Changed/unsafe artifact: "+name)
        result[str(p)]=h
    return result


def probability(p, n):
    p=np.asarray(p,dtype=float)
    if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Invalid probability")
    return p


def prepare(config_path, root=ROOT):
    path=(root/config_path).resolve(); cfg=read(path)
    if set(cfg)!={"validation_config","output_root","design"} or cfg["design"]!=DESIGN:
        raise ValueError("V1 design is frozen; do not change grids after seeing results")
    if any(not isinstance(cfg[k],str) or not cfg[k].strip() for k in ("validation_config","output_root")):
        raise ValueError("Nonempty paths required")
    output=(root/cfg["output_root"]).resolve()
    source_cfg=(root/cfg["validation_config"]).resolve()
    source=(root/read(source_cfg)["output_dir"]).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Suite output must be outside historical sources")
    from src.extensions.validation_protocol_v1 import ValidationProtocolV1
    # Existing verifier checks original versions, training data, registry, Trial14 and code.
    probe=output/"_preflight_no_write"
    c=ValidationProtocolV1()._prepare(str(source_cfg),str(probe))
    ctx=c["context"]
    if ctx["x"].shape!=(250,150): raise ValueError("Expected original250x150 data")
    hashes=dict(c["hashes"])
    hashes.update(verify_artifacts(source,{"settings.json","split_indices.json","validation_probabilities.csv","fold_scores.csv"}))
    splits=sorted([s for s in read(source/"split_indices.json") if s["scheme"]=="cv5"],key=lambda s:(s["repeat"],s["fold"]))
    expected=sorted([s for s in c["splits"] if s["scheme"]=="cv5"],key=lambda s:(s["repeat"],s["fold"]))
    if len(splits)!=50 or splits!=expected: raise ValueError("Ordered historical splits differ")
    raw=pd.read_csv(source/"validation_probabilities.csv",dtype={"id":"string"},float_precision="round_trip")
    raw=raw[raw.scheme=="cv5"]
    if len(raw)!=2500 or raw.duplicated(["repeat","fold","row_index"]).any():
        raise ValueError("Incomplete/duplicate historical probabilities")
    scores=pd.read_csv(source/"fold_scores.csv",float_precision="round_trip")
    scores=scores[scores.scheme=="cv5"]
    metadata=[]; matrices=[]; reproduction=[]; offset=0
    for s in splits:
        va=s["valid_indices"]
        part=raw[(raw.repeat==s["repeat"])&(raw.fold==s["fold"])].set_index("row_index")
        if set(part.index)!=set(va) or not (part.split_seed==s["split_seed"]).all():
            raise ValueError("Historical split coverage differs")
        part=part.loc[va]
        if not np.array_equal(part.id.astype(str),ctx["ids"].iloc[va].astype(str)) or not np.array_equal(part.target,ctx["y"].iloc[va]):
            raise ValueError("Historical IDs/labels differ")
        matrix=np.column_stack([probability(part[col],len(va)) for col in ("l1_c1","xgboost","lightgbm")])
        if not np.allclose(matrix.mean(axis=1),part.l1_xgb_lgbm,rtol=0,atol=TOL):
            raise ValueError("Original equal-weight average differs")
        # The validation table has four procedures, not native standalone tree scores.
        for alias,col,p in [("l1","l1_c1",matrix[:,0]),("ensemble","l1_xgb_lgbm",matrix.mean(axis=1))]:
            saved=scores[(scores.repeat==s["repeat"])&(scores.fold==s["fold"])&(scores.candidate==col)]
            actual=float(roc_auc_score(part.target,p))
            if len(saved)!=1 or int(saved.split_seed.iloc[0])!=s["split_seed"] or not np.isclose(actual,float(saved.auc.iloc[0]),rtol=0,atol=TOL):
                raise ValueError("Validation AUC reproduction failed: "+alias)
            reproduction.append(dict(repeat=s["repeat"],fold=s["fold"],model=alias,auc=actual,source="validation_cv5"))
        key=(s["repeat"],s["fold"])
        if c["reference_matches"].get(key) is not True:
            raise ValueError("Trial14 ordered split mismatch; cannot validate standalone trees")
        for j,alias in [(1,"xgboost"),(2,"lightgbm")]:
            actual=float(roc_auc_score(part.target,matrix[:,j]))
            expected_auc=float(c["reference"].loc[key,alias])
            if not np.isclose(actual,expected_auc,rtol=0,atol=TOL): raise ValueError("Trial14 AUC mismatch: "+alias)
            reproduction.append(dict(repeat=s["repeat"],fold=s["fold"],model=alias,auc=actual,source="verified_matching_Trial14"))
        s=dict(s,start=offset,stop=offset+len(va));offset+=len(va)
        metadata.extend(dict(repeat=s["repeat"],fold=s["fold"],split_seed=s["split_seed"],row_index=int(i),id=str(ctx["ids"].iloc[i]),target=int(ctx["y"].iloc[i])) for i in va)
        matrices.append(matrix)
    original_mean=np.mean([r["auc"] for r in reproduction if r["model"]=="ensemble"])
    if not np.isclose(original_mean,REFERENCE_AUC,rtol=0,atol=TOL): raise ValueError("Frozen reference mean mismatch")
    groups=[]; offset=0
    for s in splits:
        size=len(s["valid_indices"]);groups.append(dict(s,start=offset,stop=offset+size));offset+=size
    hashes.update({str(path):digest(path),str(Path(__file__).resolve()):digest(__file__)})
    frame=pd.DataFrame(metadata); frame["id"]=frame["id"].astype("string"); base=np.concatenate(matrices)
    cohort=sha256(frame.to_csv(index=False).encode()+base.tobytes()).hexdigest()
    signature=sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    c.update(config=cfg,output_root=output,groups=groups,frame=frame,base=base,baseline=base.mean(axis=1),
             hashes=hashes,cohort=cohort,source_signature=signature,reproduction=reproduction,suite_versions=versions())
    check_hashes(hashes)
    return c


def require_packages(task):
    import importlib
    packages={"catboost":["catboost"],"weights":[],"lgbm_es":["xgboost","lightgbm"],
              "xgb_es":["xgboost","lightgbm"],"stacking":["xgboost","lightgbm"],"combine":[]}
    for p in packages[task]: importlib.import_module(p)
    if task=="xgb_es":
        if tuple(int(v) for v in version("xgboost").split(".")[:2])<(1,6):
            raise RuntimeError("XGBoost ES requires constructor callable-metric/callback support >=1.6; do not upgrade frozen environment automatically")


def fit(model,x,y,cost,stats,**kwargs):
    stats["attempted_estimator_fits"]+=cost
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x,y,**kwargs)
    stats["warnings"].extend(dict(category=w.category.__name__,message=str(w.message)) for w in caught)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught): raise RuntimeError("ConvergenceWarning: reject partial run")
    stats["completed_estimator_fits"]+=cost
    return model


def proba(model,x):
    p=np.asarray(model.predict_proba(x),dtype=float)
    if p.shape!=(len(x),2): raise ValueError("Expected binary probability matrix")
    classes=getattr(model,"classes_",None)
    if classes is None:
        underlying=getattr(model,"model",None);classes=getattr(underlying,"classes_",[0,1])
    classes=np.asarray(classes)
    if set(classes)!={0,1}: raise ValueError("Unexpected class order")
    return probability(p[:,int(np.flatnonzero(classes==1)[0])],len(x))


def base_model(c,name):
    from src.modeling.registered_models import L1Model,XGBTopKModel,LightGBMModel
    if name=="l1": return L1Model(context=c["context"])
    if name=="xgboost": return XGBTopKModel(context=c["context"])
    if name=="lightgbm": return LightGBMModel(parameters=c["lightgbm_parameters"])
    raise ValueError(name)


def fit_base(c,name,tr,va,stats):
    ctx=c["context"];model=base_model(c,name)
    fit(model,ctx["x"].iloc[tr],ctx["y"].iloc[tr],2 if name=="xgboost" else 1,stats)
    return proba(model,ctx["x"].iloc[va])


def inner_partition(c,s):
    tr=np.asarray(s["train_indices"],dtype=int)
    y=c["context"]["y"].iloc[tr]
    seed=DESIGN["inner_seed_base"]+s["repeat"]*5+s["fold"]
    first,second=next(StratifiedShuffleSplit(n_splits=1,test_size=DESIGN["inner_fraction"],random_state=seed).split(np.zeros((len(tr),1)),y))
    return tr[first].tolist(),tr[second].tolist(),seed


def metric_factory(fixed_a,fixed_b,kind):
    def ensemble_auc(y_true,p):
        p=probability(p,len(y_true))
        value=float(roc_auc_score(y_true,(fixed_a+fixed_b+p)/3))
        return ("ensemble_auc",value,True) if kind=="lightgbm" else value
    return ensemble_auc


def select_rounds(c,s,name,stats):
    ctx=c["context"];tr,va,seed=inner_partition(c,s)
    other=[n for n in NAMES if n!=name]
    fixed=[fit_base(c,n,tr,va,stats) for n in other]
    metric=metric_factory(*fixed,name)
    cap=DESIGN["early_stop_cap"];patience=DESIGN["early_stop_patience"]
    if name=="lightgbm":
        from lightgbm import LGBMClassifier,early_stopping
        params=dict(c["lightgbm_parameters"],n_estimators=cap,metric="None")
        model=LGBMClassifier(**params)
        kwargs=dict(eval_metric=metric,callbacks=[early_stopping(patience,first_metric_only=True,verbose=False)])
        if "eval_X" in signature(model.fit).parameters:
            kwargs.update(eval_X=ctx["x"].iloc[va],eval_y=ctx["y"].iloc[va])
        else: kwargs["eval_set"]=[(ctx["x"].iloc[va],ctx["y"].iloc[va])]
        fit(model,ctx["x"].iloc[tr],ctx["y"].iloc[tr],1,stats,**kwargs)
        rounds=int(model.best_iteration_)
        metrics=model.evals_result_
        if any(set(v)!={"ensemble_auc"} for v in metrics.values()): raise ValueError("Unexpected LightGBM stopping metric")
    else:
        from xgboost import XGBClassifier
        from xgboost.callback import EarlyStopping
        from src.features.gain_top_k_selector import GainTopKSelector
        params=dict(ctx["candidate"]["estimator_parameters"])
        ranker=fit(XGBClassifier(**params),ctx["x"].iloc[tr],ctx["y"].iloc[tr],1,stats)
        selector=GainTopKSelector(k=ctx["candidate"]["k"])
        selector.fit_from_booster(ranker.get_booster(),list(ctx["x"].columns))
        # Ranker retains fixed original rounds; only reduced classifier uses ES.
        stop=EarlyStopping(rounds=patience,metric_name="ensemble_auc",data_name="validation_0",maximize=True,save_best=True)
        params.update(n_estimators=cap,eval_metric=metric,callbacks=[stop])
        model=XGBClassifier(**params)
        fit(model,selector.transform(ctx["x"].iloc[tr]),ctx["y"].iloc[tr],1,stats,
            eval_set=[(selector.transform(ctx["x"].iloc[va]),ctx["y"].iloc[va])],verbose=False)
        rounds=int(model.best_iteration)+1
        if "ensemble_auc" not in model.evals_result()["validation_0"]: raise ValueError("XGBoost custom metric missing")
    if not 1<=rounds<=cap: raise ValueError("Invalid best iteration")
    trace=dict(repeat=s["repeat"],fold=s["fold"],inner_seed=seed,inner_train_indices=tr,inner_valid_indices=va,
               selected_rounds=rounds,selected_at_round_cap=rounds==cap,selection_metric="inner_equal_weight_ensemble_auc")
    return rounds,trace


def refit_rounds(c,s,name,rounds,stats):
    ctx=c["context"];tr,va=s["train_indices"],s["valid_indices"]
    if name=="lightgbm":
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(**dict(c["lightgbm_parameters"],n_estimators=rounds))
        fit(model,ctx["x"].iloc[tr],ctx["y"].iloc[tr],1,stats)
        actual=int(model.n_estimators_)
    else:
        from xgboost import XGBClassifier
        from src.features.gain_top_k_selector import GainTopKSelector
        params=dict(ctx["candidate"]["estimator_parameters"])
        ranker=fit(XGBClassifier(**params),ctx["x"].iloc[tr],ctx["y"].iloc[tr],1,stats)
        selector=GainTopKSelector(k=ctx["candidate"]["k"])
        selector.fit_from_booster(ranker.get_booster(),list(ctx["x"].columns))
        model=XGBClassifier(**dict(params,n_estimators=rounds))
        fit(model,selector.transform(ctx["x"].iloc[tr]),ctx["y"].iloc[tr],1,stats)
        return proba(model,selector.transform(ctx["x"].iloc[va])),int(model.get_booster().num_boosted_rounds())
    return proba(model,ctx["x"].iloc[va]),actual


def make_meta():
    params=dict(C=DESIGN["stacking_meta_C"],solver="lbfgs",max_iter=5000,tol=1e-4,random_state=42)
    penalty=signature(LogisticRegression).parameters.get("penalty")
    params["l1_ratio" if penalty is None or penalty.default=="deprecated" else "penalty"]=0.0 if penalty is None or penalty.default=="deprecated" else "l2"
    return LogisticRegression(**params)


def stack_fold(c,s,stats):
    tr=np.asarray(s["train_indices"],dtype=int);ctx=c["context"]
    seed=DESIGN["stacking_seed_base"]+s["repeat"]*5+s["fold"]
    splitter=StratifiedKFold(n_splits=DESIGN["stacking_inner_folds"],shuffle=True,random_state=seed)
    oof=np.full((len(tr),3),np.nan);seen=np.zeros(len(tr),int);traces=[]
    for f,(it,iv) in enumerate(splitter.split(ctx["x"].iloc[tr],ctx["y"].iloc[tr])):
        inner_tr,inner_va=tr[it].tolist(),tr[iv].tolist()
        for j,name in enumerate(NAMES): oof[iv,j]=fit_base(c,name,inner_tr,inner_va,stats)
        seen[iv]+=1
        traces.append(dict(inner_fold=f,train_indices=inner_tr,valid_indices=inner_va))
    if not np.isfinite(oof).all() or not (seen==1).all(): raise ValueError("Invalid stacking inner OOF")
    meta=fit(make_meta(),oof,ctx["y"].iloc[tr],1,stats)
    # Cached outer reference components were fit on all outer TRAIN rows, never outer valid.
    p=proba(meta,c["base"][s["start"]:s["stop"]])
    return p,dict(repeat=s["repeat"],fold=s["fold"],inner_seed=seed,inner_splits=traces,
                  meta_coefficients=meta.coef_.tolist(),meta_intercept=meta.intercept_.tolist())


def cat_parameters():
    result=[]
    for d,it,lr,l2 in product(DESIGN["cat_depth"],DESIGN["cat_iterations"],DESIGN["cat_learning_rate"],DESIGN["cat_l2"]):
        result.append(dict(depth=d,iterations=it,learning_rate=lr,l2_leaf_reg=l2,boosting_type="Ordered",loss_function="Logloss",eval_metric="AUC",random_seed=42,task_type="CPU",thread_count=1,allow_writing_files=False,use_best_model=False,verbose=False))
    return result


def cat_fold(c,s,params,stats):
    from catboost import CatBoostClassifier
    ctx=c["context"];tr,va=s["train_indices"],s["valid_indices"]
    model=fit(CatBoostClassifier(**params),ctx["x"].iloc[tr],ctx["y"].iloc[tr],1,stats)
    return proba(model,ctx["x"].iloc[va])


def calculate(task,c,stats):
    n=len(c["frame"]);preds={};specs={};aux={};traces=[]
    if task=="weights":
        for i,w in enumerate(WEIGHTS):
            name=f"weights_{i}";preds[name]=c["baseline"].copy() if i==0 else c["base"]@np.asarray(w);specs[name]=dict(weights=list(w))
    elif task=="catboost":
        for i,params in enumerate(cat_parameters()):
            raw=np.empty(n)
            for s in c["groups"]:
                raw[s["start"]:s["stop"]]=cat_fold(c,s,params,stats)
                print(f"catboost setting={i} repeat={s['repeat']} fold={s['fold']}",flush=True)
            aux[f"cat_{i}"]=raw
            for a in ALPHAS:
                name=f"cat_{i}_alpha{a:g}"
                preds[name]=(1-a)*c["baseline"]+a*raw
                specs[name]=dict(cat_parameters=params,alpha=a,component_key=f"cat_{i}")
    elif task in ("lgbm_es","xgb_es"):
        name="lightgbm" if task=="lgbm_es" else "xgboost";j=NAMES.index(name);raw=np.empty(n)
        for s in c["groups"]:
            rounds,trace=select_rounds(c,s,name,stats)
            p,actual=refit_rounds(c,s,name,rounds,stats)
            raw[s["start"]:s["stop"]]=p;trace["actual_refit_rounds"]=actual;traces.append(trace)
            print(f"{task} repeat={s['repeat']} fold={s['fold']} rounds={rounds}",flush=True)
        base=c["base"].copy();base[:,j]=raw
        preds[task]=base.mean(axis=1);aux[name]=raw
        specs[task]=dict(replaced_component=name,other_parameters_fixed=True,final_full_data_round_policy="No deployment/refit implemented; research-only round selections saved per fold")
    elif task=="stacking":
        p=np.empty(n)
        for s in c["groups"]:
            pp,trace=stack_fold(c,s,stats);p[s["start"]:s["stop"]]=pp;traces.append(trace)
            print(f"stacking repeat={s['repeat']} fold={s['fold']}",flush=True)
        preds[task]=p;specs[task]=dict(meta_C=DESIGN["stacking_meta_C"],inner_folds=DESIGN["stacking_inner_folds"])
    else: raise ValueError("Unsupported training task")
    return preds,specs,aux,traces


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
    details=[];comparisons=[]
    ref=repeats[repeats.candidate=="reference"].set_index("repeat").mean_fold_auc.sort_index()
    if list(ref.index)!=list(range(10)): raise ValueError("Exactly ten reference repetitions required")
    for name in preds:
        if name=="reference": continue
        current=repeats[repeats.candidate==name].set_index("repeat").mean_fold_auc.sort_index()
        if not current.index.equals(ref.index): raise ValueError("Unpaired candidate")
        delta=current-ref
        details.extend(dict(candidate=name,repeat=int(r),difference=float(d)) for r,d in delta.items())
        comparisons.append(dict(reference="reference",candidate=name,mean_difference=float(delta.mean()),difference_sd=float(delta.std(ddof=1)),wins=int((delta>TOL).sum()),losses=int((delta< -TOL).sum()),ties=int((abs(delta)<=TOL).sum())))
    return folds,repeats,summary,pd.DataFrame(details),pd.DataFrame(comparisons)


def decision_for(summary,comparison,specs):
    means=summary.set_index("candidate").mean_auc
    selected=max(specs,key=lambda n:means[n])
    stat=comparison[comparison.candidate==selected].iloc[0].to_dict()
    return dict(selected_candidate=selected,selected_mean_auc=float(means[selected]),selected_spec=specs[selected],
                screening_passed=bool(stat["mean_difference"]>TOL and stat["wins"]>=6),automatic_promotion=False,
                selection_rule="Highest ensemble mean; exact ties use preregistered generation order. No fallback.",**stat,
                interpretation="Adaptive development research, not nested/independent evaluation or a significance test. Inner validation only prevents direct stopping/meta-training leakage; prior adaptivity remains.")


def load_child(c,task):
    root=c["output_root"]/task
    hashes=verify_artifacts(root,{"settings.json","decision.json","cohort.csv","predictions.npz","model_summary.csv","comparison.csv","run_manifest.json"}-{"run_manifest.json"})
    settings=read(root/"settings.json")
    if settings["source_signature"]!=c["source_signature"] or settings["cohort"]!=c["cohort"] or settings["design"]!=DESIGN:
        raise ValueError("Combination sources use different cohort/design/reference")
    actual=pd.read_csv(root/"cohort.csv",dtype={"id":"string"},float_precision="round_trip")
    expected_frame=c["frame"].copy(); expected_frame["id"]=expected_frame["id"].astype("string")
    if not actual.equals(expected_frame): raise ValueError("Combination cohort identity mismatch")
    with np.load(root/"predictions.npz",allow_pickle=False) as z: arrays={n:z[n].copy() for n in z.files}
    for p in arrays.values(): probability(p,len(c["frame"]))
    specs=settings["candidate_specs"]
    _,_,ss,_,comp=score(c,{"reference":c["baseline"],**{n:arrays["prediction__"+n] for n in specs}})
    recomputed=decision_for(ss,comp,specs)
    decision=read(root/"decision.json")
    if recomputed!=decision: raise ValueError("Combination source decision failed reproduction")
    return decision,arrays,hashes


def combine(c):
    children={};hashes={}
    for task in ("catboost","weights","lgbm_es","xgb_es"):
        d,arrays,h=load_child(c,task);children[task]=(d,arrays);hashes.update(h)
    passing=[task for task,(d,_) in children.items() if d["screening_passed"]]
    if len(passing)<2: return {},{}, {},[dict(skipped="Fewer than two individually screening-passed changes",passing=passing)],hashes
    base=c["base"].copy();weights=WEIGHTS[0];alpha=0;cat=None
    if "weights" in passing: weights=children["weights"][0]["selected_spec"]["weights"]
    for task,name,j in (("lgbm_es","lightgbm",2),("xgb_es","xgboost",1)):
        if task in passing: base[:,j]=children[task][1]["component__"+name]
    p=base@np.asarray(weights) if "weights" in passing else base.mean(axis=1)
    if "catboost" in passing:
        d,z=children["catboost"];spec=d["selected_spec"];alpha=spec["alpha"];cat=z["component__"+spec["component_key"]]
        p=(1-alpha)*p+alpha*cat
    # Validate exactly ONE preregistered all-passing combination; no subset/weight search.
    return {"all_passing_combined":p},{"all_passing_combined":dict(included_changes=passing,weights=list(weights),cat_alpha=alpha)}, {},[],hashes


def plan():
    return dict(reference="Frozen L1/XGBoost-top20/LightGBM Trial14; no rolling reference",
                tasks=dict(catboost=dict(settings=16,ensemble_candidates=48,estimator_fits=800),weights=dict(candidates=4,estimator_fits=0),
                lgbm_es=dict(candidates=1,estimator_fits=250),xgb_es=dict(candidates=1,estimator_fits=300),
                combine=dict(candidates="one all-passing combination or skip",estimator_fits=0),stacking=dict(candidates=1,estimator_fits=650)),
                total_planned_estimator_fits=2000,screening="Positive paired mean >1e-12 and >=6/10 wins; not automatic promotion",
                final_refit_or_submission_implemented=False)


def run_task(c,task,dry_run=False):
    dest=c["output_root"]/task
    if dest.exists(): raise FileExistsError("Fresh per-experiment output required: "+str(dest))
    require_packages(task);check_hashes(c["hashes"])
    if task=="combine": payload=combine(c)
    else: payload=None
    if dry_run:
        return dict(status="preflight_passed",task=task,fits_performed=0,test_opened=False,output=str(dest),plan=plan()["tasks"][task])
    dest.mkdir(parents=True,exist_ok=False)
    stats=dict(attempted_estimator_fits=0,completed_estimator_fits=0,warnings=[])
    manifest=dict(status="running",task=task,test_opened=False,kaggle_upload_performed=False,started_utc=datetime.now(timezone.utc).isoformat())
    write(dest/"run_manifest.json",manifest)
    hashes=dict(c["hashes"])
    try:
        with threadpool_limits(limits=1), (dest/"fit.log").open("w",encoding="utf-8") as log:
            with redirect_stdout(log),redirect_stderr(log):
                if payload is None: preds,specs,aux,traces=calculate(task,c,stats)
                else: preds,specs,aux,traces,more_hashes=payload;hashes.update(more_hashes)
        c["frame"].to_csv(dest/"cohort.csv",index=False)
        write(dest/"settings.json",dict(task=task,design=DESIGN,source_signature=c["source_signature"],cohort=c["cohort"],
              input_and_code_sha256=hashes,versions=c["suite_versions"],candidate_specs=specs))
        write(dest/"split_indices.json",c["groups"]);write(dest/"inner_traces.json",traces)
        pd.DataFrame(c["reproduction"]).to_csv(dest/"source_reproduction.csv",index=False)
        if not preds:
            write(dest/"decision.json",dict(status="skipped",automatic_promotion=False,reason=traces))
            manifest["status"]="skipped"
        else:
            for p in list(preds.values())+list(aux.values()): probability(p,len(c["frame"]))
            folds,repeats,summary,paired,comparison=score(c,{"reference":c["baseline"],**preds})
            decision=decision_for(summary,comparison,specs)
            for name,df in (("fold_scores",folds),("repeat_scores",repeats),("model_summary",summary),("paired_repeat_differences",paired),("comparison",comparison)):
                df.to_csv(dest/(name+".csv"),index=False)
            np.savez_compressed(dest/"predictions.npz",**{"prediction__"+n:p for n,p in preds.items()},**{"component__"+n:p for n,p in aux.items()})
            write(dest/"decision.json",decision)
            if aux:
                _,_,component_summary,_,_=score(c,{"reference":c["baseline"],**aux})
                component_summary.to_csv(dest/"component_summary.csv",index=False)
            manifest["status"]="completed"
        check_hashes(hashes)
        write(dest/"warnings.json",stats["warnings"])
        manifest.update({k:v for k,v in stats.items() if k!="warnings"})
        manifest["artifact_sha256"]={p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!="run_manifest.json"}
        return dict(status=manifest["status"],task=task,output=str(dest),automatic_promotion=False,**{k:v for k,v in stats.items() if k!="warnings"})
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}")
        write(dest/"warnings.json",stats["warnings"])
        raise
    finally:
        manifest.update({k:v for k,v in stats.items() if k!="warnings"})
        manifest["finished_utc"]=datetime.now(timezone.utc).isoformat()
        write(dest/"run_manifest.json",manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["plan","preflight","run"])
    p.add_argument("--config",default="configs/ensemble_research_suite_v1.json")
    p.add_argument("--experiment",choices=list(TASKS)+["all"],default="all")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    if args.command=="plan": print(json.dumps(plan(),indent=2));return
    c=prepare(args.config)
    tasks=list(TASKS) if args.experiment=="all" else [args.experiment]
    for t in tasks: require_packages(t)
    if args.command=="preflight":
        if tasks==["combine"]: combine(c)
        for t in tasks:
            if (c["output_root"]/t).exists(): raise FileExistsError("Fresh output required for "+t)
        print(json.dumps(dict(status="preflight_passed",tasks=tasks,fits_performed=0,test_opened=False,plan=plan()),indent=2));return
    for t in tasks:
        # all --dry-run has no earlier child outputs; validate combine dependencies only when run alone.
        if args.dry_run and t=="combine" and args.experiment=="all":
            print(json.dumps(dict(task=t,status="deferred_dependency_check",reason="Checked at combine execution after four completed source experiments")));continue
        print(json.dumps(run_task(c,t,args.dry_run),indent=2),flush=True)


if __name__=="__main__": main()

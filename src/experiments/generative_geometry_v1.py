"""Geometry-first generative experiment. No competition-test reader or export path."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from src.features.generative_geometry_v1 import GeometryProjector
from src.modeling.semi_supervised_mixture_v1 import SemiSupervisedMixture

ROOT=Path(__file__).resolve().parents[2]
DESIGN=dict(clusters=[2,4,8],covariances=["tied","diag"],initialization_seeds=[42,314,2026],
    selected_features=20,latent_dimensions=8,unlabelled_weight=1.,reg_covar=.001,
    label_smoothing=.001,max_iter=300,tol=.0001)
TOL=1e-12


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def check(hashes):
    for p,h in hashes.items():
        if digest(p)!=h:raise ValueError("Changed source: "+str(p))


def prepare(config_path,output=None,transductive_validation=False):
    cfgpath=(ROOT/config_path).resolve();cfg=read(cfgpath)
    required={"train_path","target_column","id_column","expected_rows","expected_features","reference_study","output_dir","design"}
    # Extra test_path/permission fields are rejected BEFORE opening any input data.
    if set(cfg)!=required or cfg["design"]!=DESIGN:
        raise ValueError("Unknown config or changed frozen design. Competition-test mode is NOT implemented")
    train=(ROOT/cfg["train_path"]).resolve();source=(ROOT/cfg["reference_study"]).resolve()
    out=(ROOT/(output or cfg["output_dir"])).resolve()
    if out.exists():raise FileExistsError("Fresh output required")
    if out.is_relative_to(source) or source.is_relative_to(out) or train.is_relative_to(out):raise ValueError("Output overlaps source")
    from src.data.labelled_data_loader import LabelledDataLoader
    x,y,ids,features=LabelledDataLoader().load(SimpleNamespace(train_path=train,id_column=cfg["id_column"],target_column=cfg["target_column"],
        expected_rows=cfg["expected_rows"],expected_features=cfg["expected_features"]))
    manifest=read(source/"run_manifest.json");artifacts=manifest.get("artifact_sha256",{})
    needed={"settings.json","split_indices.json","cohort.csv","decision.json","trials/trial_0025/predictions.npz"}
    if manifest.get("status")!="completed" or manifest.get("completed_trials")!=30 or not needed<=artifacts.keys():raise ValueError("Completed30-trial reference with required hashes needed")
    hashes={str(cfgpath):digest(cfgpath),str(train):digest(train),str(source/"run_manifest.json"):digest(source/"run_manifest.json")}
    for name,h in artifacts.items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or digest(path)!=h:raise ValueError("Unsafe/changed reference artifact: "+name)
        hashes[str(path)]=h
    settings=read(source/"settings.json")
    bound=settings.get("input_and_code_sha256",{})
    if bound.get(str(train))!=digest(train):raise ValueError("Reference does not bind this exact labelled CSV")
    decision=read(source/"decision.json")
    if decision.get("selected_trial")!=25 or decision.get("control_reproduced") is not True:raise ValueError("Frozen Trial25 reference required")
    groups=read(source/"split_indices.json")
    frame=pd.read_csv(source/"cohort.csv",dtype={"id":"string"},float_precision="round_trip")
    if len(groups)!=50 or len(frame)!=10*len(x):raise ValueError("Expected50 folds and ten OOF appearances per row")
    if [(s["repeat"],s["fold"]) for s in groups]!=list(product(range(10),range(5))):raise ValueError("Invalid fold order")
    counts=np.zeros((10,len(x)),dtype=int);offset=0
    for s in groups:
        tr=np.asarray(s["train_indices"]);va=np.asarray(s["valid_indices"])
        if tr.dtype.kind not in "iu" or va.dtype.kind not in "iu":raise ValueError("Noninteger indices")
        if len(set(tr))!=len(tr) or len(set(va))!=len(va) or set(tr)&set(va) or set(tr)|set(va)!=set(range(len(x))):raise ValueError("Invalid split partition")
        if set(y.iloc[tr])!={0,1} or set(y.iloc[va])!={0,1}:raise ValueError("Both classes required")
        if s["start"]!=offset or s["stop"]!=offset+len(va):raise ValueError("Invalid prediction offsets")
        part=frame.iloc[s["start"]:s["stop"]]
        if not np.array_equal(part.row_index,va) or not np.array_equal(part.id.astype(str),ids.iloc[va].astype(str)) or not np.array_equal(part.target,y.iloc[va]) or not (part.repeat==s["repeat"]).all() or not (part.fold==s["fold"]).all():raise ValueError("Reference cohort identity mismatch")
        counts[s["repeat"],va]+=1;offset+=len(va)
    if not (counts==1).all():raise ValueError("Every row must be validated exactly once per repetition")
    with np.load(source/"trials/trial_0025/predictions.npz",allow_pickle=False) as z:reference=np.asarray(z["ensemble"],dtype=float)
    probability(reference,len(frame))
    reference_auc=np.mean([roc_auc_score(frame.target.iloc[s["start"]:s["stop"]],reference[s["start"]:s["stop"]]) for s in groups])
    if not np.isclose(reference_auc,decision["selected_mean_auc"],rtol=0,atol=TOL):raise ValueError("Reference AUC failed reproduction")
    from src.features import generative_geometry_v1
    from src.modeling import semi_supervised_mixture_v1
    import inspect
    for module in [generative_geometry_v1,semi_supervised_mixture_v1]:
        p=Path(inspect.getfile(module)).resolve();hashes[str(p)]=digest(p)
    hashes[str(Path(__file__).resolve())]=digest(__file__)
    loader=Path(inspect.getfile(LabelledDataLoader)).resolve();hashes[str(loader)]=digest(loader)
    check(hashes)
    return dict(config=cfg,x=x,y=y,ids=ids,features=features,groups=groups,frame=frame,reference=reference,
        reference_auc=float(reference_auc),output=out,hashes=hashes,modes=["A","B"] if transductive_validation else ["A"])


def probability(p,n):
    if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("Invalid probability")


def evaluate(c,predictions):
    rows=[]
    for name,p in predictions.items():
        probability(p,len(c["frame"]))
        for s in c["groups"]:
            sl=slice(s["start"],s["stop"])
            rows.append(dict(candidate=name,repeat=s["repeat"],fold=s["fold"],auc=float(roc_auc_score(c["frame"].target.iloc[sl],p[sl]))))
    folds=pd.DataFrame(rows);repeats=folds.groupby(["candidate","repeat"],as_index=False).auc.mean().rename(columns={"auc":"mean_fold_auc"})
    summary=repeats.groupby("candidate").mean_fold_auc.agg(mean_auc="mean",repeat_sd="std").reset_index()
    return folds,repeats,summary


def compare(repeats,candidate,reference):
    get=lambda n:repeats[repeats.candidate==n].set_index("repeat").mean_fold_auc.sort_index()
    a,b=get(candidate),get(reference)
    if not a.index.equals(b.index):raise ValueError("Unpaired comparisons")
    d=a-b
    return dict(candidate=candidate,reference=reference,mean_difference=float(d.mean()),difference_sd=float(d.std(ddof=1)),
        wins=int((d>TOL).sum()),losses=int((d< -TOL).sum()),ties=int((abs(d)<=TOL).sum()))


def execute(c):
    out=c["output"];out.mkdir(parents=True,exist_ok=False)
    specs={f"{mode}_k{k}_{cov}":dict(mode=mode,n_components=k,covariance_type=cov) for mode,k,cov in product(c["modes"],DESIGN["clusters"],DESIGN["covariances"])}
    manifest=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),competition_test_opened=False,kaggle_upload_performed=False,
        transductive_validation="B" in c["modes"],planned_candidate_fits=len(specs)*len(c["groups"])*len(DESIGN["initialization_seeds"]),completed_candidate_fits=0,
        fit_accounting="Each candidate fit includes one sklearn GMM initialization plus one custom EM run; projector fits counted separately",completed_projector_fits=0)
    write(out/"run_manifest.json",manifest);predictions={name:np.full(len(c["frame"]),np.nan) for name in specs}
    diagnostics=[];geometry=[];eligible={name:True for name in specs}
    try:
        check(c["hashes"])
        write(out/"settings.json",dict(config=c["config"],modes=c["modes"],input_and_code_sha256=c["hashes"],versions={p:version(p) for p in ["numpy","scipy","pandas","scikit-learn","threadpoolctl"]},
            scope="A uses training features only. B ALSO uses validation features, NEVER validation labels. No competition-test support.",
            selection="Highest mean AUC among fully converged seed-averaged candidates; exact ties generation order. No seed selection, no blending, no auto-promotion."))
        c["frame"].to_csv(out/"cohort.csv",index=False);write(out/"split_indices.json",c["groups"])
        with threadpool_limits(limits=1), (out/"fit_diagnostics.jsonl").open("w",encoding="utf-8") as stream:
            for s in c["groups"]:
                tr,va=s["train_indices"],s["valid_indices"];xl=c["x"].iloc[tr].to_numpy();xv=c["x"].iloc[va].to_numpy();yl=c["y"].iloc[tr].to_numpy()
                for mode in c["modes"]:
                    fit_x=xl if mode=="A" else np.vstack([xl,xv])
                    projector=GeometryProjector(DESIGN["selected_features"],DESIGN["latent_dimensions"]).fit(fit_x)
                    geometry.append(dict(repeat=s["repeat"],fold=s["fold"],mode=mode,**projector.diagnostics_));manifest["completed_projector_fits"]+=1
                    zl,zv=projector.transform(xl),projector.transform(xv)
                    for name,spec in specs.items():
                        if spec["mode"]!=mode:continue
                        seed_predictions=[]
                        for seed in DESIGN["initialization_seeds"]:
                            model=SemiSupervisedMixture(n_components=spec["n_components"],covariance_type=spec["covariance_type"],
                                unlabelled_weight=DESIGN["unlabelled_weight"],reg_covar=DESIGN["reg_covar"],label_smoothing=DESIGN["label_smoothing"],
                                max_iter=DESIGN["max_iter"],tol=DESIGN["tol"],random_state=seed).fit(zl,yl,zv if mode=="B" else None)
                            p=model.predict_proba(zv)[:,1];probability(p,len(va));seed_predictions.append(p)
                            eligible[name]=eligible[name] and bool(model.converged_)
                            row=dict(candidate=name,repeat=s["repeat"],fold=s["fold"],seed=seed,converged=bool(model.converged_),n_iter=model.n_iter_,
                                initialization_warnings=model.initialization_warnings_,effective_masses=model.effective_masses_,class_probabilities=model.class_probabilities_.tolist(),
                                objective_trace=model.objective_trace_,objective_decreases=model.objective_decreases_)
                            stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush()
                            manifest["completed_candidate_fits"]+=1
                        arr=np.array(seed_predictions);predictions[name][s["start"]:s["stop"]]=arr.mean(axis=0)
                        diagnostics.append(dict(candidate=name,repeat=s["repeat"],fold=s["fold"],mean_seed_prediction_sd=float(arr.std(axis=0).mean())))
                write(out/"run_manifest.json",manifest)
                print(f"fold={s['repeat']}/{s['fold']} fits={manifest['completed_candidate_fits']}/{manifest['planned_candidate_fits']}",flush=True)
        folds,repeats,summary=evaluate(c,{"reference_trial25":c["reference"],**predictions})
        comparisons=[compare(repeats,name,"reference_trial25") for name in specs]
        mechanism=[compare(repeats,name,name.replace("B_","A_",1)) for name in specs if name.startswith("B_")]
        means=summary.set_index("candidate").mean_auc
        valid=[n for n in specs if eligible[n]];selected=max(valid,key=lambda n:means[n]) if valid else None
        stat=next((v for v in comparisons if v["candidate"]==selected),None)
        decision=dict(selected_candidate=selected,selected_mean_auc=float(means[selected]) if selected else None,
            selected_spec=specs[selected] if selected else None,eligible_candidates=eligible,
            screening_passed=bool(stat and stat["mean_difference"]>TOL and stat["wins"]>=6),reference_comparison=stat,automatic_promotion=False,
            interpretation="Exploratory reused-label development evidence. B is transductive, not untouched inductive CV. No claim of competition permission, independent evaluation, or statistical significance.")
        for name,df in [("fold_scores",folds),("repeat_scores",repeats),("model_summary",summary),("comparison",pd.DataFrame(comparisons)),
                       ("B_vs_A",pd.DataFrame(mechanism)),("seed_stability",pd.DataFrame(diagnostics))]:df.to_csv(out/(name+".csv"),index=False)
        write(out/"geometry_diagnostics.json",geometry);write(out/"candidate_specs.json",specs);write(out/"decision.json",decision)
        np.savez_compressed(out/"predictions.npz",reference_trial25=c["reference"],**predictions)
        check(c["hashes"]);manifest["status"]="completed"
        manifest["artifact_sha256"]={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!="run_manifest.json"}
        return decision
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        manifest["finished_utc"]=datetime.now(timezone.utc).isoformat();write(out/"run_manifest.json",manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/generative_geometry_v1.json")
    parser.add_argument("--output");parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--transductive-validation",action="store_true",help="Explicitly enable B: validation FEATURES participate; no competition-test access")
    args=parser.parse_args();c=prepare(args.config,args.output,args.transductive_validation)
    if args.dry_run:
        print(json.dumps(dict(status="preflight_passed",modes=c["modes"],candidate_fits=len(c["modes"])*900,projector_fits=len(c["modes"])*50,
            fits_performed=0,competition_test_opened=False,reference_auc=c["reference_auc"],output=str(c["output"])),indent=2))
    else:print(json.dumps(execute(c),indent=2))

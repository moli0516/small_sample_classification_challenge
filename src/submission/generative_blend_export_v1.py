"""Read-only fixed GMM/Trial25 50:50 export. No fitting, pickle loading or upload."""
import argparse
import csv
from datetime import datetime,timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
TOL=1e-12
NAME="blend_gmm0.5"
EXPECTED_CV_AUC=.9382748290356986
PARAMETERS=dict(C=.18702703609689042,l1_ratio=.9589771634360093,alpha=.06655791394873416)
DESIGN=dict(gmm_weight=.5,trial25_weight=.5)


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def path(p):return (ROOT/p).resolve()
def close(a,b,label):
    if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a,b,rtol=0,atol=TOL):raise ValueError("Mismatch: "+label)
def probability(p,n):
    p=np.asarray(p,dtype=float)
    if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("Invalid probabilities")
    return p

def merge(target,more):
    for p,h in more.items():
        p=str(Path(p).resolve())
        if not isinstance(h,str) or not re.fullmatch(r"[0-9a-f]{64}",h):raise ValueError("Invalid SHA256: "+p)
        if p in target and target[p]!=h:raise ValueError("Conflicting source hashes: "+p)
        target[p]=h

def check(hashes):
    for p,h in hashes.items():
        if not Path(p).is_file() or digest(p)!=h:raise ValueError("Changed/missing source: "+p)


def artifact_bundle(root,manifest_name,required):
    root=Path(root).resolve();mp=root/manifest_name;m=read(mp);hashes={str(mp):digest(mp)}
    table=m.get("artifact_sha256",{})
    if not isinstance(table,dict) or not required<=table.keys():raise ValueError("Required artifact hashes missing: "+str(root))
    for name,h in table.items():
        p=(root/name).resolve()
        if Path(name).is_absolute() or not p.is_relative_to(root) or p==mp or not p.is_file():raise ValueError("Unsafe/missing artifact: "+name)
        merge(hashes,{str(p):h})
    check(hashes);return m,hashes


def csv_frame(p,expected_columns=None,ids_only=False):
    with Path(p).open(encoding="utf-8-sig",newline="") as stream:header=next(csv.reader(stream),[])
    if not header or len(header)!=len(set(header)) or any(not s.strip() for s in header):raise ValueError("Invalid/duplicate CSV headers: "+str(p))
    if "id" not in header or (expected_columns is not None and header!=expected_columns):raise ValueError("CSV column mismatch: "+str(p))
    df=pd.read_csv(p,dtype={"id":"string"},float_precision="round_trip",usecols=["id"] if ids_only else None)
    ids=df.id
    if ids.isna().any() or not ids.is_unique or ids.str.strip().eq("").any():raise ValueError("Empty/duplicate IDs: "+str(p))
    return df


def ordered_ids_sha(ids):return sha256(json.dumps(list(map(str,ids)),ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def score(frame,groups,p):
    p=probability(p,len(frame));values=[]
    for s in groups:
        sl=slice(s["start"],s["stop"])
        values.append(dict(repeat=s["repeat"],fold=s["fold"],auc=float(roc_auc_score(frame.target.iloc[sl],p[sl]))))
    return pd.DataFrame(values).groupby("repeat").auc.mean().sort_index()


def comparison(current,reference):
    d=current-reference
    return dict(mean_difference=float(d.mean()),difference_sd=float(d.std(ddof=1)),wins=int((d>TOL).sum()),losses=int((d< -TOL).sum()),ties=int((abs(d)<=TOL).sum()))


def verify_cv(root):
    needed={"settings.json","decision.json","cohort.csv","split_indices.json","predictions.npz","candidate_specs.json"}
    m,hashes=artifact_bundle(root,"run_manifest.json",needed)
    if m.get("status")!="completed" or m.get("action")!="blend" or m.get("planned_candidate_fits")!=0 or m.get("completed_candidate_fits")!=0 or m.get("competition_test_opened") is not False:raise ValueError("Expected completed zero-fit/no-test blending study")
    decision=read(root/"decision.json");settings=read(root/"settings.json")
    if decision.get("action")!="blend" or decision.get("selected_candidate")!=NAME or decision.get("selected_spec")!=DESIGN or decision.get("screening_against_both_controls") is not True or decision.get("automatic_promotion") is not False:raise ValueError("Expected fixed screening-passed blend_gmm0.5")
    merge(hashes,settings["input_and_code_sha256"])
    frame=pd.read_csv(root/"cohort.csv",dtype={"id":"string"},float_precision="round_trip");groups=read(root/"split_indices.json")
    if len(frame)!=2500 or len(groups)!=50 or [(s["repeat"],s["fold"]) for s in groups]!=[(r,f) for r in range(10) for f in range(5)]:raise ValueError("Invalid CV coverage")
    if frame.id.isna().any() or set(frame.row_index)!=set(range(250)) or set(frame.target)!={0,1}:raise ValueError("Invalid CV identity/labels")
    if not (frame.groupby("row_index").id.nunique()==1).all() or not (frame.groupby("row_index").target.nunique()==1).all() or frame[["row_index","id"]].drop_duplicates().id.nunique()!=250:raise ValueError("CV row/ID/label identity changes")
    offset=0;counts=np.zeros((10,250),dtype=int)
    for s in groups:
        tr,va=s["train_indices"],s["valid_indices"]
        if any(type(i) is not int for i in tr+va) or len(set(tr))!=len(tr) or len(set(va))!=len(va) or set(tr)&set(va) or set(tr)|set(va)!=set(range(250)):raise ValueError("Invalid split partition")
        if s["start"]!=offset or s["stop"]!=offset+len(va):raise ValueError("Invalid CV offsets")
        part=frame.iloc[s["start"]:s["stop"]]
        if not np.array_equal(part.row_index,va) or not (part.repeat==s["repeat"]).all() or not (part.fold==s["fold"]).all():raise ValueError("CV row order mismatch")
        counts[s["repeat"],va]+=1;offset+=len(va)
    if not (counts==1).all():raise ValueError("Incomplete CV coverage")
    with np.load(root/"predictions.npz",allow_pickle=False) as z:
        arrays={key:probability(z[key],len(frame)).copy() for key in ["A_k8_tied","reference_trial25",*[f"blend_gmm{w:g}" for w in [.25,.5,.75]]]}
    stats={};means={}
    for w in [.25,.5,.75]:
        name=f"blend_gmm{w:g}";expected=w*arrays["A_k8_tied"]+(1-w)*arrays["reference_trial25"]
        if not np.allclose(arrays[name],expected,rtol=0,atol=TOL):raise ValueError("CV blend formula mismatch")
        stats[name]=score(frame,groups,expected);means[name]=float(stats[name].mean())
    if max(means,key=means.get)!=NAME:raise ValueError("Highest-mean CV rule does not select0.5")
    close(means[NAME],decision["selected_mean_auc"],"CV decision AUC");close(means[NAME],EXPECTED_CV_AUC,"frozen CV AUC")
    controls={key:score(frame,groups,arrays[key]) for key in ["A_k8_tied","reference_trial25"]}
    for key,current in controls.items():
        rows=[v for v in decision["comparisons"] if v["reference"]==key and v["candidate"]==NAME]
        if len(rows)!=1:raise ValueError("Missing CV control comparison")
        actual=comparison(stats[NAME],current)
        for k,v in actual.items():close(v,rows[0][k],"CV comparison "+k)
        if actual["mean_difference"]<=TOL or actual["wins"]<6:raise ValueError("CV screening did not reproduce")
    specs=read(root/"candidate_specs.json")
    if specs.get(NAME)!=DESIGN:raise ValueError("Saved blend specification differs")
    check(hashes)
    return dict(decision=decision,settings=settings,hashes=hashes,control_aucs={k:float(v.mean()) for k,v in controls.items()})


def verify_export(root,kind,expected_rows):
    manifest_name="run_manifest.json" if kind=="gmm" else "prediction_manifest.json"
    required={"submission.csv","settings.json","model_bundle.joblib"} if kind=="gmm" else {"submission.csv","component_predictions.csv","ensemble.json","source_decision.json","source_best_trial.json"}
    m,hashes=artifact_bundle(root,manifest_name,required)
    if m.get("status")!="ready_for_manual_submission" or m.get("saved_and_reloaded_models_verified") is not True or m.get("rows")!=expected_rows or m.get("kaggle_upload_performed") is not False:raise ValueError("Source export is not verified/ready")
    if kind=="gmm":
        if m.get("action")!="export" or m.get("planned_candidate_fits")!=3 or m.get("completed_candidate_fits")!=3 or m.get("competition_test_used_for_fitting") is not False:raise ValueError("Expected training-only three-seed K8 export")
        settings=read(root/"settings.json");merge(hashes,settings["input_and_code_sha256"])
        d=settings["source_design"]
        if d["initialization_seeds"]!=[42,314,2026] or d["selected_features"]!=20 or d["latent_dimensions"]!=8:raise ValueError("GMM representation/seed recipe differs")
        inputs=m["export_input_sha256"]
    else:
        if m.get("candidate")!="elasticnet_optuna_trial_25" or m.get("selected_trial")!=25 or m.get("parameters")!=PARAMETERS or m.get("fits_performed")!=5:raise ValueError("Expected frozen Trial25 export")
        close(float(m["submission_sha256"]==digest(root/"submission.csv")),1.,"Trial25 submission hash")
        best=read(root/"source_best_trial.json");decision=read(root/"source_decision.json");settings=read(root/"ensemble.json")
        if best.get("number")!=25 or best.get("state")!="COMPLETE" or best.get("parameters")!=PARAMETERS or decision.get("selected_trial")!=25 or decision.get("parameters")!=PARAMETERS or decision.get("screening_both_controls") is not True:raise ValueError("Trial25 source decision/parameters differ")
        if settings.get("candidate")!="elasticnet_optuna_trial_25" or settings.get("elasticnet_parameters")!=PARAMETERS:raise ValueError("Trial25 recipe differs")
        close(settings["alpha"],PARAMETERS["alpha"],"Trial25 alpha")
        merge(hashes,m["source_input_and_code_sha256"]);inputs=m["prediction_input_sha256"]
    merge(hashes,inputs);check(hashes)
    df=csv_frame(root/"submission.csv",["id","target"]);probability(df.target.to_numpy(),expected_rows)
    return dict(manifest=m,hashes=hashes,inputs=inputs,settings=settings,frame=df)


def prepare(config_path,output=None):
    cfgpath=path(config_path);cfg=read(cfgpath)
    if set(cfg)!={"gmm_export","trial25_export","blend_study","train_path","test_path","sample_submission_path","expected_rows","output_dir","design"} or cfg["design"]!=DESIGN or type(cfg["expected_rows"]) is not int or cfg["expected_rows"]<=0:raise ValueError("Fixed50/50 config required")
    roots={k:path(cfg[k]) for k in ["gmm_export","trial25_export","blend_study"]};dest=path(output or cfg["output_dir"])
    if dest.exists():raise FileExistsError("Fresh output required; no overwrite/resume")
    for root in roots.values():
        if dest.is_relative_to(root) or root.is_relative_to(dest):raise ValueError("Output overlaps source")
    if len(set(roots.values()))!=3:raise ValueError("Three distinct source directories required")
    official={k:path(cfg[k]) for k in ["train_path","test_path","sample_submission_path"]}
    if any(p.is_relative_to(dest) for p in official.values()):raise ValueError("Output overlaps input")
    hashes={str(cfgpath):digest(cfgpath),str(Path(__file__).resolve()):digest(__file__)}
    cli=ROOT/"scripts/export_generative_blend_v1.py";hashes[str(cli)]=digest(cli)
    official_hashes={str(p):digest(p) for p in official.values()};merge(hashes,official_hashes)
    cv=verify_cv(roots["blend_study"]);merge(hashes,cv["hashes"])
    sources={key:verify_export(roots[key],kind,cfg["expected_rows"]) for key,kind in [("gmm_export","gmm"),("trial25_export","trial25")]}
    for item in sources.values():merge(hashes,item["hashes"])
    trainhash=official_hashes[str(official["train_path"])];testhash=official_hashes[str(official["test_path"])];templatehash=official_hashes[str(official["sample_submission_path"])]
    for key,item in sources.items():
        if item["hashes"].get(str(official["train_path"]))!=trainhash:raise ValueError("Source does not bind exact official training CSV: "+key)
        if testhash not in item["inputs"].values() or templatehash not in item["inputs"].values():raise ValueError("Source export used different test/template: "+key)
    gs=sources["gmm_export"]["settings"];cs=cv["settings"]
    if Path(gs["source_study"]).resolve()!=Path(cs["source_study"]).resolve() or gs["source_design"]!=cs["source_design"]:raise ValueError("GMM export and CV study have different generative lineage")
    gmm_source=Path(gs["source_study"]).resolve();decision_path=gmm_source/"decision.json"
    if dest.is_relative_to(gmm_source) or gmm_source.is_relative_to(dest):raise ValueError("Output overlaps original GMM study")
    original_trial=sources["trial25_export"]["manifest"].get("source_study")
    if original_trial:
        original_trial=Path(original_trial).resolve()
        if dest.is_relative_to(original_trial) or original_trial.is_relative_to(dest):raise ValueError("Output overlaps original Trial25 study")
    if str(decision_path) not in gs["input_and_code_sha256"]:raise ValueError("GMM source decision is not hash-bound")
    gd=read(decision_path)
    if gd.get("selected_candidate")!="A_k8_tied" or gd.get("selected_spec")!=dict(mode="A",n_components=8,covariance_type="tied") or not gd.get("eligible_candidates",{}).get("A_k8_tied") or gd.get("screening_passed") is not True:raise ValueError("GMM source is not eligible selected K8 tied")
    close(gd["selected_mean_auc"],cv["control_aucs"]["A_k8_tied"],"GMM source/CV score")
    close(read(roots["trial25_export"]/"source_best_trial.json")["objective_auc"],cv["control_aucs"]["reference_trial25"],"Trial25 source/CV score")
    test=csv_frame(official["test_path"],ids_only=True);template=csv_frame(official["sample_submission_path"],["id","target"])
    if len(test)!=cfg["expected_rows"] or len(template)!=cfg["expected_rows"] or set(test.id)!=set(template.id):raise ValueError("Official test/template ID set mismatch")
    aligned={};alignment={}
    for key,item in sources.items():
        df=item["frame"]
        if len(df)!=len(template) or set(df.id)!=set(template.id):raise ValueError("Source submission ID set mismatch: "+key)
        aligned[key]=df.set_index("id").loc[template.id,"target"].to_numpy()
        alignment[key]=dict(original_order_sha256=ordered_ids_sha(df.id),template_order_sha256=ordered_ids_sha(template.id),reordered_to_template=not df.id.reset_index(drop=True).equals(template.id.reset_index(drop=True)))
    # Trial25 audit is originally in test order. Reconstruct its original formula by ID.
    audit=csv_frame(roots["trial25_export"]/"component_predictions.csv")
    required={"id","l1","xgboost","lightgbm","elasticnet","original_three_model","ensemble"}
    if not required<=set(audit.columns) or len(audit)!=len(test) or set(audit.id)!=set(test.id):raise ValueError("Trial25 component audit coverage differs")
    for col in required-{"id"}:probability(audit[col].to_numpy(),len(test))
    raw=(audit.l1.to_numpy()+audit.xgboost.to_numpy()+audit.lightgbm.to_numpy())/3
    reconstructed=(1-PARAMETERS["alpha"])*raw+PARAMETERS["alpha"]*audit.elasticnet.to_numpy()
    if not np.allclose(raw,audit.original_three_model,rtol=0,atol=TOL) or not np.allclose(reconstructed,audit.ensemble,rtol=0,atol=TOL):raise ValueError("Trial25 component formula did not reproduce")
    by_id=pd.Series(reconstructed,index=audit.id).loc[template.id].to_numpy()
    if not np.allclose(by_id,aligned["trial25_export"],rtol=0,atol=TOL):raise ValueError("Trial25 audit/submission mismatch")
    alignment["trial25_component_audit"]=dict(matches_official_test_order=audit.id.reset_index(drop=True).equals(test.id.reset_index(drop=True)),original_order_sha256=ordered_ids_sha(audit.id))
    blend=.5*aligned["gmm_export"]+.5*aligned["trial25_export"];probability(blend,len(template));check(hashes)
    return dict(config=cfg,output=dest,roots=roots,hashes=hashes,sources=sources,cv=cv,template=template,
        aligned=aligned,alignment=alignment,blend=blend,official_hashes=official_hashes)


def export(c):
    out=c["output"];out.mkdir(parents=True,exist_ok=False)
    m=dict(status="running",candidate=NAME,weights=DESIGN,started_utc=datetime.now(timezone.utc).isoformat(),fits_performed=0,
        competition_test_opened=True,competition_test_used_for_fitting=False,kaggle_upload_performed=False,automatic_promotion=False,
        source_export_directories={k:str(v) for k,v in c["roots"].items()},source_input_and_code_sha256=c["hashes"],official_input_sha256=c["official_hashes"],
        versions={p:version(p) for p in ["numpy","pandas","scikit-learn"]})
    write(out/"prediction_manifest.json",m)
    final=out/"submission.csv";pending=out/"submission.pending.csv"
    try:
        check(c["hashes"])
        write(out/"settings.json",c["config"]);write(out/"source_cv_decision.json",c["cv"]["decision"])
        write(out/"lineage.json",dict(candidate=NAME,formula="0.5*p_A_k8_tied+0.5*p_elasticnet_optuna_trial_25",weights=DESIGN,
            alignment=c["alignment"],sources={k:dict(directory=str(c["roots"][k]),manifest_sha256=digest(c["roots"][k]/("run_manifest.json" if k=="gmm_export" else "prediction_manifest.json")),submission_sha256=digest(c["roots"][k]/"submission.csv")) for k in c["sources"]},
            verification_scope="Recorded export lineage/artifact integrity and Trial25 formula verified. GMM model bytes hashed, NOT deserialized or rerun. No model fitting or automatic upload."))
        frame=pd.DataFrame({"id":c["template"].id.to_numpy(),"target":c["blend"]})
        pd.DataFrame({"id":frame.id,"gmm_k8":c["aligned"]["gmm_export"],"trial25":c["aligned"]["trial25_export"],"blend":c["blend"]}).to_csv(out/"aligned_source_predictions.csv",index=False,float_format="%.17g")
        frame.to_csv(pending,index=False,float_format="%.17g")
        reread=csv_frame(pending,["id","target"]);probability(reread.target.to_numpy(),len(frame))
        if not np.array_equal(reread.id.astype(str),frame.id.astype(str)) or not np.allclose(reread.target,frame.target,rtol=0,atol=TOL):raise ValueError("Submission round-trip failed")
        check(c["hashes"]);pending.replace(final)
        m.update(status="ready_for_manual_submission",rows=len(frame),submission_sha256=digest(final),csv_roundtrip_verified=True,
            source_predictions_verified=True,prediction_min=float(frame.target.min()),prediction_max=float(frame.target.max()))
        m["artifact_sha256"]={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!="prediction_manifest.json"}
        return dict(status=m["status"],candidate=NAME,rows=len(frame),fits_performed=0,output=str(out),kaggle_upload_performed=False)
    except BaseException as exc:
        final.unlink(missing_ok=True);pending.unlink(missing_ok=True)
        m.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        m["finished_utc"]=datetime.now(timezone.utc).isoformat();write(out/"prediction_manifest.json",m)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/generative_blend_export_v1.json");p.add_argument("--output");p.add_argument("--dry-run",action="store_true")
    args=p.parse_args();c=prepare(args.config,args.output)
    if args.dry_run:print(json.dumps(dict(status="preflight_passed",candidate=NAME,rows=len(c["template"]),fits_performed=0,competition_test_opened=True,competition_test_used_for_fitting=False,output_created=False,output=str(c["output"]),alignment=c["alignment"]),indent=2))
    else:print(json.dumps(export(c),indent=2))

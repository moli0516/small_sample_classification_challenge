"""Trial21 zero-fit weight comparison and read-only selected-candidate export."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.submission import generative_trial21_snapshot_v1 as source

ROOT=Path(__file__).resolve().parents[2]
b=source.b
TOL=1e-12
DESIGN=dict(alphas=[.5,.75,.9,.95])


def specs():
    result={"new_gmm":dict(family="control",alpha=1.,formula="G_new",effective_weights=dict(new_gmm=1.,old_gmm=0.,trial25=0.))}
    for family in ["A","B"]:
        for a in DESIGN["alphas"]:
            result[f"{family}_a{a:g}"]=dict(family=family,alpha=a,formula="alpha*G_new+(1-alpha)*"+("H_old" if family=="A" else "T"),effective_weights=dict(new_gmm=a,old_gmm=(1-a)/2 if family=="A" else 0.,trial25=(1-a)/2 if family=="A" else 1-a))
    return result


def calculate(c):
    ss=specs();predictions={"old_gmm":c["control"],"old_hybrid":c["hybrid_control"],"trial25":c["reference"]}
    for name,spec in ss.items():
        a=spec["alpha"]
        predictions[name]=c["new_gmm"].copy() if spec["family"]=="control" else a*c["new_gmm"]+(1-a)*(c["hybrid_control"] if spec["family"]=="A" else c["reference"])
    if not np.allclose(predictions["B_a0.5"],c["new_hybrid"],rtol=0,atol=TOL):raise ValueError("H_new/B_a0.5 did not reproduce")
    folds,repeats,summary=b.evaluate(c,predictions);means=summary.set_index("candidate").mean_auc
    comparisons=[b.compare(repeats,name,ref) for name in ss for ref in ["new_gmm","old_hybrid","B_a0.5"] if name!=ref]
    decision=select(means,comparisons)
    return predictions,folds,repeats,summary,pd.DataFrame(comparisons),decision


def select(means,comparisons):
    ss=specs();selected=max(ss,key=lambda n:means[n]);control=selected=="new_gmm"
    stats=[v for v in comparisons if v["candidate"]==selected]
    get=lambda name:next(v for v in stats if v["reference"]==name)
    passed=lambda v:bool(v["mean_difference"]>TOL and v["wins"]>=6)
    primary=None if control else passed(get("new_gmm"));retained=selected if primary else "new_gmm"
    return dict(selected_candidate=selected,selected_spec=ss[selected],selected_mean_auc=float(means[selected]),selected_is_control=control,
        screening_passed=primary,screening_status="not_applicable_control_selected" if control else ("passed" if primary else "failed"),
        old_hybrid_screening_passed=passed(get("old_hybrid")),retained_candidate=retained,retained_spec=ss[retained],retained_mean_auc=float(means[retained]),
        comparisons=stats,automatic_promotion=False,aliases={"H_new":"B_a0.5"},
        selection_rule="Highest mean; ties standalone then A alpha order then B alpha order. If winning blend fails, retain standalone, never another blend.",
        interpretation="Post-search fixed-grid development comparison on reused labels/splits; no independent evaluation, significance claim or public-score tuning")


def prepare(config_path,action,output=None):
    cfgpath=(ROOT/config_path).resolve();cfg=b.read(cfgpath)
    required={"trial21_export_config","trial21_export","old_hybrid_export","comparison_output","export_output","design"}
    if set(cfg)!=required or cfg["design"]!=DESIGN:raise ValueError("Fixed weight comparison config required")
    if action not in ("compare","export"):raise ValueError("Unknown action")
    dest=(ROOT/(output or cfg["comparison_output" if action=="compare" else "export_output"])).resolve()
    if dest.exists():raise FileExistsError("Fresh output required; no overwrite/resume")
    roots=[(ROOT/cfg[k]).resolve() for k in ["trial21_export","old_hybrid_export","comparison_output"]]
    for root in roots:
        if action=="compare" and root==(ROOT/cfg["comparison_output"]).resolve():continue
        if dest.is_relative_to(root) or root.is_relative_to(dest):raise ValueError("Output overlaps source")
    c=source.prepare(str((ROOT/cfg["trial21_export_config"]).resolve()),output=str(dest))
    directory=c["study_root"]/"trials/trial_0021"
    with np.load(directory/"predictions.npz",allow_pickle=False) as z:
        new=np.asarray(z["gmm"],dtype=float).copy();hybrid=np.asarray(z["hybrid"],dtype=float).copy()
    for p in [new,hybrid]:b.probability(p,len(c["frame"]))
    if not np.allclose(c["hybrid_control"],.5*c["control"]+.5*c["reference"],rtol=0,atol=TOL):raise ValueError("H_old formula differs")
    watched=dict(c["hashes"])
    source.merge(watched,{str(p.resolve()):b.digest(p) for p in [cfgpath,Path(__file__).resolve(),ROOT/"scripts/run_trial21_weights_v1.py"]})
    b.check(watched);c.update(output=dest,weights_config=cfg,hashes=watched,new_gmm=new,new_hybrid=hybrid,action=action)
    return c


def finish(c,m,manifest_name):
    m["finished_utc"]=datetime.now(timezone.utc).isoformat()
    if m["status"] in ("completed","ready_for_manual_submission"):
        m["artifact_sha256"]={str(p.relative_to(c["output"])):b.digest(p) for p in c["output"].rglob("*") if p.is_file() and p.name!=manifest_name}
    b.write(c["output"]/manifest_name,m)


def compare(c):
    out=c["output"];out.mkdir(parents=True,exist_ok=False)
    m=dict(status="running",action="compare",fits_performed=0,competition_test_opened=False,kaggle_upload_performed=False,started_utc=datetime.now(timezone.utc).isoformat())
    b.write(out/"run_manifest.json",m)
    try:
        b.check(c["hashes"]);preds,folds,repeats,summary,paired,decision=calculate(c)
        b.write(out/"settings.json",dict(config=c["weights_config"],input_and_code_sha256=c["hashes"],candidate_specs=specs()))
        c["frame"].to_csv(out/"cohort.csv",index=False);b.write(out/"split_indices.json",c["groups"])
        b.write(out/"candidate_specs.json",specs())
        pd.DataFrame([dict(candidate=n,family=v["family"],alpha=v["alpha"],formula=v["formula"],**{k+"_weight":z for k,z in v["effective_weights"].items()}) for n,v in specs().items()]).to_csv(out/"candidate_specs.csv",index=False)
        for name,df in [("fold_scores",folds),("repeat_scores",repeats),("model_summary",summary),("paired_comparisons",paired)]:df.to_csv(out/(name+".csv"),index=False)
        details=[]
        for row in paired.to_dict("records"):
            get=lambda name:repeats[repeats.candidate==name].set_index("repeat").mean_fold_auc.sort_index()
            delta=get(row["candidate"])-get(row["reference"])
            details.extend(dict(candidate=row["candidate"],reference=row["reference"],repeat=int(i),difference=float(v)) for i,v in delta.items())
        pd.DataFrame(details).to_csv(out/"paired_repeat_differences.csv",index=False)
        np.savez_compressed(out/"predictions.npz",**preds);b.write(out/"decision.json",decision)
        b.check(c["hashes"]);m["status"]="completed";return decision
    except BaseException as exc:
        m.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}");(out/"decision.json").unlink(missing_ok=True);raise
    finally:finish(c,m,"run_manifest.json")


def frozen_comparison(c):
    root=(ROOT/c["weights_config"]["comparison_output"]).resolve()
    m,watched=source.artifacts(root,"run_manifest.json",{"settings.json","decision.json","cohort.csv","split_indices.json","predictions.npz","model_summary.csv"},"completed")
    if m.get("action")!="compare" or m.get("fits_performed")!=0 or m.get("competition_test_opened") is not False:raise ValueError("Expected completed zero-fit comparison")
    settings=b.read(root/"settings.json")
    if settings["config"]!=c["weights_config"] or settings["candidate_specs"]!=specs():raise ValueError("Frozen comparison config changed")
    b.check(settings["input_and_code_sha256"])
    expected=c["frame"].copy();expected["id"]=expected.id.astype("string")
    actual=pd.read_csv(root/"cohort.csv",dtype={"id":"string"},float_precision="round_trip")
    if not actual.equals(expected) or b.read(root/"split_indices.json")!=c["groups"]:raise ValueError("Comparison cohort/splits differ")
    predictions,_,_,_,_,replayed=calculate(c);decision=b.read(root/"decision.json")
    if replayed!=decision:raise ValueError("Comparison decision failed exact reproduction")
    with np.load(root/"predictions.npz",allow_pickle=False) as z:
        if set(z.files)!=set(predictions) or any(not np.allclose(z[n],p,rtol=0,atol=TOL) for n,p in predictions.items()):raise ValueError("Comparison prediction formula differs")
    source.merge(c["hashes"],watched);source.merge(c["hashes"],settings["input_and_code_sha256"])
    b.check(c["hashes"]);return decision


def aligned(root,relative,template,n):
    df=source.csv_frame(root/relative,["id","target"]);b.probability(df.target.to_numpy(),n)
    if len(df)!=n or set(df.id)!=set(template.id):raise ValueError("Source prediction ID coverage differs")
    return df.set_index("id").loc[template.id,"target"].to_numpy()


def check_prediction_inputs(c,declared,current):
    if any(h not in declared.values() for h in current.values()):raise ValueError("Source used different test/template contents")
    b.check(declared);source.merge(c["hashes"],declared)


def test_predictions(c,decision):
    cfg=c["weights_config"];root=(ROOT/cfg["trial21_export"]).resolve();n=c["export_config"]["expected_rows"]
    required={"standalone/submission.csv","standalone/candidate_manifest.json","hybrid/submission.csv","settings.json","source_best_trial.json","source_decision.json","aligned_predictions.csv","model_bundle.joblib"}
    m,watched=source.artifacts(root,"prediction_manifest.json",required,"ready_for_manual_submission")
    if m.get("selected_trial")!=21 or m.get("parameters")!=source.PARAMETERS or m.get("completed_candidate_fits")!=3 or m.get("saved_and_reloaded_models_verified") is not True or m.get("rows")!=n or m.get("competition_test_used_for_fitting") is not False or m.get("kaggle_upload_performed") is not False:raise ValueError("Wrong/unverified Trial21 export")
    if b.read(root/"source_best_trial.json")!=c["source_best"] or b.read(root/"source_decision.json")!=c["source_decision"]:raise ValueError("Trial21 export uses different selected source")
    b.check(m["source_input_and_code_sha256"]);source.merge(c["hashes"],watched);source.merge(c["hashes"],m["source_input_and_code_sha256"])
    settings=b.read(root/"settings.json")
    if settings["parameters"]!=source.PARAMETERS or settings["seeds"]!=[42,314,2026] or settings["covariance"]!="tied":raise ValueError("Trial21 export recipe differs")
    paths={k:(source.ROOT/c["export_config"][k]).resolve() for k in ["test_path","sample_submission_path"]}
    current={str(p):b.digest(p) for p in paths.values()};check_prediction_inputs(c,m["prediction_input_sha256"],current)
    # IDs only; no test feature learning and no model loading.
    with paths["test_path"].open(encoding="utf-8-sig",newline="") as stream:
        import csv
        header=next(csv.reader(stream),[])
    if len(header)!=len(set(header)) or "id" not in header:raise ValueError("Bad test header")
    ids=pd.read_csv(paths["test_path"],usecols=["id"],dtype={"id":"string"}).id
    template=source.csv_frame(paths["sample_submission_path"],["id","target"])
    if len(ids)!=n or len(template)!=n or ids.isna().any() or not ids.is_unique or ids.str.strip().eq("").any() or set(ids)!=set(template.id):raise ValueError("Official test/template ID coverage differs")
    values={"new_gmm":aligned(root,"standalone/submission.csv",template,n)};lineage={"new_gmm":str(root)}
    audit=source.csv_frame(root/"aligned_predictions.csv")
    if len(audit)!=n or set(audit.id)!=set(template.id):raise ValueError("Trial21 audit IDs differ")
    audit=audit.set_index("id").loc[template.id]
    for name in ["trial21_gmm","trial25","trial21_hybrid"]:b.probability(audit[name].to_numpy(),n)
    if not np.allclose(audit.trial21_gmm,values["new_gmm"],rtol=0,atol=TOL) or not np.allclose(audit.trial21_hybrid,.5*audit.trial21_gmm+.5*audit.trial25,rtol=0,atol=TOL):raise ValueError("Trial21 source audit formula differs")
    if not np.allclose(aligned(root,"hybrid/submission.csv",template,n),audit.trial21_hybrid,rtol=0,atol=TOL):raise ValueError("Trial21 source hybrid differs")
    w=decision["retained_spec"]["effective_weights"]
    if w["trial25"]>0:
        tr=c["trial25_root"];tm,th=source.artifacts(tr,"prediction_manifest.json",{"submission.csv","component_predictions.csv","source_best_trial.json"},"ready_for_manual_submission")
        if tm.get("selected_trial")!=25 or tm.get("parameters")!=source.TRIAL25_PARAMETERS or tm.get("saved_and_reloaded_models_verified") is not True:raise ValueError("Wrong Trial25 export")
        source.merge(c["hashes"],th);source.merge(c["hashes"],tm["source_input_and_code_sha256"]);check_prediction_inputs(c,tm["prediction_input_sha256"],current)
        values["trial25"]=aligned(tr,"submission.csv",template,n)
        if not np.allclose(values["trial25"],audit.trial25,rtol=0,atol=TOL):raise ValueError("Trial25 differs from Trial21 frozen source")
        parts=source.csv_frame(tr/"component_predictions.csv")
        if len(parts)!=n or set(parts.id)!=set(template.id):raise ValueError("Trial25 audit coverage differs")
        parts=parts.set_index("id").loc[template.id]
        for col in ["l1","xgboost","lightgbm","elasticnet","original_three_model","ensemble"]:b.probability(parts[col].to_numpy(),n)
        old=(parts.l1+parts.xgboost+parts.lightgbm)/3;reconstructed=(1-source.TRIAL25_PARAMETERS["alpha"])*old+source.TRIAL25_PARAMETERS["alpha"]*parts.elasticnet
        if not np.allclose(old,parts.original_three_model,rtol=0,atol=TOL) or not np.allclose(reconstructed,parts.ensemble,rtol=0,atol=TOL) or not np.allclose(reconstructed,values["trial25"],rtol=0,atol=TOL):raise ValueError("Trial25 formula differs")
        lineage["trial25"]=str(tr)
    if w["old_gmm"]>0:
        oldroot=(ROOT/cfg["old_hybrid_export"]).resolve()
        hm,hh=source.artifacts(oldroot,"prediction_manifest.json",{"submission.csv","aligned_source_predictions.csv","lineage.json","source_cv_decision.json"},"ready_for_manual_submission")
        if hm.get("candidate")!="blend_gmm0.5" or hm.get("weights")!={"gmm_weight":.5,"trial25_weight":.5} or hm.get("fits_performed")!=0:raise ValueError("Wrong old hybrid export")
        source.merge(c["hashes"],hh);source.merge(c["hashes"],hm["source_input_and_code_sha256"]);check_prediction_inputs(c,hm["official_input_sha256"],current)
        oldcv=b.read(oldroot/"source_cv_decision.json")
        _,_,ss=b.evaluate(c,{"old_hybrid":c["hybrid_control"]});source.close(oldcv["selected_mean_auc"],float(ss.mean_auc.iloc[0]),"old hybrid CV source")
        if oldcv.get("selected_candidate")!="blend_gmm0.5":raise ValueError("Old hybrid CV candidate differs")
        oldline=b.read(oldroot/"lineage.json")
        if oldline["sources"]["trial25_export"]["submission_sha256"]!=b.digest(c["trial25_root"]/"submission.csv"):raise ValueError("Old hybrid uses another Trial25 export")
        parts=source.csv_frame(oldroot/"aligned_source_predictions.csv")
        if len(parts)!=n or set(parts.id)!=set(template.id):raise ValueError("Old hybrid audit coverage differs")
        parts=parts.set_index("id").loc[template.id]
        for name in ["gmm_k8","trial25","blend"]:b.probability(parts[name].to_numpy(),n)
        if not np.allclose(parts.trial25,values["trial25"],rtol=0,atol=TOL) or not np.allclose(parts.blend,.5*parts.gmm_k8+.5*parts.trial25,rtol=0,atol=TOL) or not np.allclose(parts.blend,aligned(oldroot,"submission.csv",template,n),rtol=0,atol=TOL):raise ValueError("Old hybrid source formula differs")
        gmroot=Path(oldline["sources"]["gmm_export"]["directory"]).resolve()
        gm,gh=source.artifacts(gmroot,"run_manifest.json",{"settings.json","submission.csv","model_bundle.joblib"},"ready_for_manual_submission")
        if gm.get("action")!="export" or gm.get("completed_candidate_fits")!=3 or gm.get("saved_and_reloaded_models_verified") is not True:raise ValueError("Unverified original K8 export")
        gs=b.read(gmroot/"settings.json")
        if Path(gs["source_study"]).resolve()!=Path(c["source_study"]).resolve() or gs["source_design"]!=b.DESIGN:raise ValueError("Original K8 CV/export lineage differs")
        if oldline["sources"]["gmm_export"]["submission_sha256"]!=b.digest(gmroot/"submission.csv") or oldline["sources"]["gmm_export"]["manifest_sha256"]!=b.digest(gmroot/"run_manifest.json"):raise ValueError("Old hybrid K8 fingerprint differs")
        source.merge(c["hashes"],gh);source.merge(c["hashes"],gs["input_and_code_sha256"]);check_prediction_inputs(c,gm["export_input_sha256"],current)
        if not np.allclose(parts.gmm_k8,aligned(gmroot,"submission.csv",template,n),rtol=0,atol=TOL):raise ValueError("Old hybrid K8 audit differs")
        values["old_gmm"]=parts.gmm_k8.to_numpy();values["old_hybrid"]=parts.blend.to_numpy();lineage["old_gmm"]=str(gmroot);lineage["old_hybrid"]=str(oldroot)
    chosen=decision["retained_spec"];a=chosen["alpha"]
    p=values["new_gmm"].copy() if chosen["family"]=="control" else a*values["new_gmm"]+(1-a)*values["old_hybrid" if chosen["family"]=="A" else "trial25"]
    b.probability(p,n);source.merge(c["hashes"],current);b.check(c["hashes"])
    return template,p,values,lineage


def export(c,decision):
    out=c["output"];out.mkdir(parents=True,exist_ok=False)
    m=dict(status="running",action="export",candidate=decision["retained_candidate"],fits_performed=0,competition_test_opened=False,competition_test_used_for_fitting=False,kaggle_upload_performed=False,automatic_promotion=False,started_utc=datetime.now(timezone.utc).isoformat())
    b.write(out/"prediction_manifest.json",m);final=out/"submission.csv";pending=out/"submission.pending.csv"
    try:
        b.check(c["hashes"]);m["competition_test_opened"]=True;b.write(out/"prediction_manifest.json",m)
        template,p,values,lineage=test_predictions(c,decision)
        b.write(out/"source_decision.json",decision);b.write(out/"lineage.json",dict(candidate=decision["retained_candidate"],spec=decision["retained_spec"],sources=lineage))
        pd.DataFrame({"id":template.id.to_numpy(),**values,"selected":p}).to_csv(out/"aligned_predictions.csv",index=False,float_format="%.17g")
        pd.DataFrame({"id":template.id.to_numpy(),"target":p}).to_csv(pending,index=False,float_format="%.17g")
        reread=source.csv_frame(pending,["id","target"]);b.probability(reread.target.to_numpy(),len(template))
        if not np.array_equal(reread.id.astype(str),template.id.astype(str)) or not np.allclose(reread.target,p,rtol=0,atol=TOL):raise ValueError("Submission round-trip failed")
        b.check(c["hashes"]);pending.replace(final)
        m.update(status="ready_for_manual_submission",rows=len(template),csv_roundtrip_verified=True,source_predictions_verified=True,submission_sha256=b.digest(final),source_input_and_code_sha256=c["hashes"])
        return dict(status=m["status"],candidate=m["candidate"],rows=len(template),fits_performed=0,output=str(out),automatic_upload=False)
    except BaseException as exc:
        final.unlink(missing_ok=True);pending.unlink(missing_ok=True);m.update(status="failed_or_interrupted",error=f"{type(exc).__name__}: {exc}");raise
    finally:finish(c,m,"prediction_manifest.json")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["compare","export"]);p.add_argument("--config",default="configs/trial21_weights_v1.json");p.add_argument("--output");p.add_argument("--dry-run",action="store_true")
    args=p.parse_args();c=prepare(args.config,args.action,args.output);decision=frozen_comparison(c) if args.action=="export" else None
    if args.action=="export":
        required=[(ROOT/c["weights_config"]["trial21_export"])/"prediction_manifest.json",(ROOT/c["weights_config"]["trial21_export"])/"standalone/submission.csv"]
        if decision["retained_spec"]["effective_weights"]["old_gmm"]>0:required.append((ROOT/c["weights_config"]["old_hybrid_export"])/"prediction_manifest.json")
        for file in required:
            if not file.is_file():raise FileNotFoundError("Finish existing source export first: "+str(file))
    if args.dry_run:
        print(json.dumps(dict(status="preflight_passed",action=args.action,fits_performed=0,competition_test_opened=False,output_created=False,retained_candidate=decision["retained_candidate"] if decision else None,prediction_input_validation="deferred" if args.action=="export" else "not_used",export_artifact_content_validation="deferred" if args.action=="export" else "not_used",output=str(c["output"])),indent=2))
    else:print(json.dumps(compare(c) if args.action=="compare" else export(c,decision),indent=2))

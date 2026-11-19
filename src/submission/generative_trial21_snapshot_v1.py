"""Frozen Trial21 source verification; dry-run never opens competition inputs."""
import csv
from pathlib import Path
from importlib.metadata import version
import inspect
import json
import numpy as np
import pandas as pd
from src.experiments import generative_optuna_v1 as search

ROOT=Path(__file__).resolve().parents[2]
b=search.b
TOL=1e-12
PARAMETERS=dict(selected_features=20,latent_dimensions=12,n_components=12,reg_covar=1.4159458366980159e-05,label_smoothing=1.3711160213584192e-05)
GMM_AUC=.9954691845996194
HYBRID_AUC=.9837300018821757
TRIAL25_PARAMETERS=dict(C=.18702703609689042,l1_ratio=.9589771634360093,alpha=.06655791394873416)
PINS={"best_trial.json":"49ed6e1e72e7625c78d152638af52aead7adee2fb655f4dd93fbddfb683de977",
      "decision.json":"5af7fe9b4956d955f584fd1f88d2b0c2c7ddcc8b49e7b1f6b4216adc265d638b",
      "trials/trial_0021/parameters.json":"787d523e9c5b7926cd38bbb1a00c26686af50721259309e0c6613e51a706ad21",
      "trials/trial_0021/predictions.npz":"f06c5fe9e34c76c1dfdc7f4a3a4fd5c3f42a3f0e517bc62da35014bcb8ba31ef"}


def close(a,z,label):
    if not np.isfinite(a) or not np.isfinite(z) or not np.isclose(a,z,rtol=0,atol=TOL):raise ValueError("Mismatch: "+label)
def merge(target,more):
    for p,h in more.items():
        p=str(Path(p).resolve())
        if p in target and target[p]!=h:raise ValueError("Conflicting source hash: "+p)
        target[p]=h

def artifacts(root,manifest_name,required,status):
    root=Path(root).resolve();mp=root/manifest_name;m=b.read(mp);hashes={str(mp):b.digest(mp)}
    if m.get("status")!=status or not required<=m.get("artifact_sha256",{}).keys():raise ValueError("Incomplete/unready source: "+str(root))
    for name,h in m["artifact_sha256"].items():
        p=(root/name).resolve()
        if Path(name).is_absolute() or not p.is_relative_to(root) or p==mp or not p.is_file():raise ValueError("Unsafe artifact: "+name)
        merge(hashes,{str(p):h})
    b.check(hashes);return m,hashes


def csv_frame(p,columns=None):
    with Path(p).open(encoding="utf-8-sig",newline="") as stream:header=next(csv.reader(stream),[])
    if not header or len(set(header))!=len(header) or any(not s.strip() for s in header) or "id" not in header or (columns and header!=columns):raise ValueError("Invalid CSV schema/header: "+str(p))
    df=pd.read_csv(p,dtype={"id":"string"},float_precision="round_trip")
    if df.id.isna().any() or not df.id.is_unique or df.id.str.strip().eq("").any():raise ValueError("Missing/duplicate IDs")
    return df


def verify_result(c,root,number,result):
    directory=root/"trials"/f"trial_{number:04d}"
    if result["number"]!=number or result["state"]!="COMPLETE" or b.read(directory/"parameters.json")!=result["parameters"]:raise ValueError("Trial result/parameter identity mismatch")
    with np.load(directory/"predictions.npz",allow_pickle=False) as z:
        raw=np.asarray(z["gmm"],dtype=float).copy();hybrid=np.asarray(z["hybrid"],dtype=float).copy()
    b.probability(raw,len(c["frame"]));b.probability(hybrid,len(c["frame"]))
    if not np.allclose(hybrid,.5*raw+.5*c["reference"],rtol=0,atol=TOL):raise ValueError("Saved hybrid formula differs")
    preds=dict(candidate_gmm=raw,candidate_hybrid=hybrid,original_gmm=c["control"],original_hybrid=c["hybrid_control"],reference_trial25=c["reference"])
    _,rr,ss=b.evaluate(c,preds);means=ss.set_index("candidate").mean_auc
    close(means["candidate_hybrid"],result["objective_auc"],"replayed hybrid");close(means["candidate_gmm"],result["standalone_auc"],"replayed standalone")
    if number==0:
        if result["parameters"]!=search.CONTROL or result.get("control_reproduced") is not True:raise ValueError("Control declaration differs")
        if not np.allclose(raw,c["control"],rtol=0,atol=TOL) or not np.allclose(hybrid,c["hybrid_control"],rtol=0,atol=TOL):raise ValueError("Control row-level reproduction differs")
    else:
        if result["parameters"]!=PARAMETERS:raise ValueError("Not frozen Trial21")
        close(means["candidate_gmm"],GMM_AUC,"Trial21 standalone");close(means["candidate_hybrid"],HYBRID_AUC,"Trial21 hybrid")
    pairs=[("candidate_hybrid",n) for n in ["original_hybrid","original_gmm","reference_trial25"]]+[("candidate_gmm","original_gmm")]
    for current,ref in pairs:
        actual=b.compare(rr,current,ref);recorded=[v for v in result["comparisons"] if v["candidate"]==current and v["reference"]==ref]
        if len(recorded)!=1:raise ValueError("Missing trial comparison")
        for k in ["mean_difference","difference_sd","wins","losses","ties"]:close(actual[k],recorded[0][k],"paired "+k)
    recorded=pd.read_csv(directory/"model_summary.csv",float_precision="round_trip").set_index("candidate")
    for row in ss.to_dict("records"):
        for k in ["mean_auc","repeat_sd"]:close(row[k],recorded.loc[row["candidate"],k],"model summary "+k)
    fits=[json.loads(line) for line in (directory/"fit_diagnostics.jsonl").read_text().splitlines() if line.strip()]
    wanted={(s["repeat"],s["fold"],seed) for s in c["groups"] for seed in b.DESIGN["initialization_seeds"]}
    if len(fits)!=len(wanted) or {(v["repeat"],v["fold"],v["seed"]) for v in fits}!=wanted or any(v.get("converged") is not True or v.get("error") is not None for v in fits):raise ValueError("Trial seed coverage/convergence invalid")
    geometry=[json.loads(line) for line in (directory/"geometry_diagnostics.jsonl").read_text().splitlines() if line.strip()]
    expected_keys={(v["repeat"],v["fold"]) for v in c["groups"]}
    if len(geometry)!=len(expected_keys) or {(v["repeat"],v["fold"]) for v in geometry}!=expected_keys:raise ValueError("Geometry fold coverage differs")
    params=result["parameters"]
    training_sizes={(v["repeat"],v["fold"]):len(v["train_indices"]) for v in c["groups"]}
    for v in geometry:
        if v["requested_features"]!=params["selected_features"] or v["requested_dimensions"]!=params["latent_dimensions"] or not 1<=v["effective_features"]<=params["selected_features"]:raise ValueError("Geometry request differs")
        expected_dimension=min(params["latent_dimensions"],v["effective_features"],training_sizes[(v["repeat"],v["fold"]) ]-1)
        if v["effective_dimensions"]!=expected_dimension or len(set(v["selected_indices"]))!=v["effective_features"] or any(type(i) is not int or not 0<=i<c["x"].shape[1] for i in v["selected_indices"]):raise ValueError("Geometry effective dimension/columns differ")
    return ss


def prepare(config_path,output=None):
    cfgpath=(ROOT/config_path).resolve();cfg=b.read(cfgpath)
    required={"study","study_config","trial25_export","test_path","sample_submission_path","expected_rows","output_dir"}
    if set(cfg)!=required or type(cfg["expected_rows"]) is not int or cfg["expected_rows"]<=0:raise ValueError("Exact Trial21 export config required")
    dest=(ROOT/(output or cfg["output_dir"])).resolve();root=(ROOT/cfg["study"]).resolve();trialroot=(ROOT/cfg["trial25_export"]).resolve()
    if dest.exists():raise FileExistsError("Fresh export output required")
    for source in [root,trialroot]:
        if dest.is_relative_to(source) or source.is_relative_to(dest):raise ValueError("Output overlaps source")
    # Uses the local frozen source verifier, but does no fitting or competition-test I/O.
    c=search.snapshot.prepare(str((ROOT/cfg["study_config"]).resolve()),output=str(dest))
    if c["x"].shape!=(250,150) or len(c["groups"])!=50 or len(c["frame"])!=2500:raise ValueError("Original250x150/50-fold source required")
    if c["config"].get("id_column")!="id" or c["config"].get("target_column")!="target":raise ValueError("Explicit id,target schema required")
    hashes=dict(c["hashes"])
    needed={"settings.json","decision.json","best_trial.json","trials.json","trials.csv","cohort.csv","split_indices.json"}
    for number in [0,21]:
        needed.update(f"trials/trial_{number:04d}/{name}" for name in ["parameters.json","result.json","predictions.npz","model_summary.csv","fit_diagnostics.jsonl","geometry_diagnostics.jsonl"])
    m,h=artifacts(root,"run_manifest.json",needed,"completed");merge(hashes,h)
    counts=dict(planned_trials=30,attempted_trials=30,completed_trials=30,rejected_trials=0,failed_trials=0,
        planned_candidate_fits=4500,attempted_candidate_fits=4500,completed_candidate_fits=4500,converged_candidate_fits=4500,attempted_projector_fits=1500,completed_projector_fits=1500)
    if any(m.get(k)!=v for k,v in counts.items()) or m.get("competition_test_opened") is not False or m.get("kaggle_upload_performed") is not False:raise ValueError("Not the completed30-trial/4500-fit no-test study")
    for name,h in PINS.items():
        if b.digest(root/name)!=h:raise ValueError("Frozen study fingerprint differs: "+name)
    settings=b.read(root/"settings.json")
    if settings["config"]!=c["optuna_config"] or settings["fixed_source_design"]!=b.DESIGN or settings["hybrid_weights"]!=[.5,.5]:raise ValueError("Source evaluated recipe differs")
    b.check(settings["input_and_code_sha256"]);merge(hashes,settings["input_and_code_sha256"])
    for package,expected in dict(settings["versions"],optuna=settings["optuna_version"]).items():
        if version(package)!=expected:raise ValueError("Source version changed: "+package)
    frame=pd.read_csv(root/"cohort.csv",dtype={"id":"string"},float_precision="round_trip");expected=c["frame"].copy();expected["id"]=expected.id.astype("string")
    if not frame.equals(expected) or b.read(root/"split_indices.json")!=c["groups"]:raise ValueError("Source cohort/splits differ")
    trials=b.read(root/"trials.json")
    if len(trials)!=30 or [t["number"] for t in trials]!=list(range(30)) or any(t["state"]!="COMPLETE" or not np.isfinite(t["value"]) for t in trials):raise ValueError("Incomplete trial accounting")
    winner=min(trials,key=lambda t:(-t["value"],t["number"]))
    if winner["number"]!=21 or winner["parameters"]!=PARAMETERS or trials[0]["parameters"]!=search.CONTROL or trials[0]["user_attrs"].get("control_reproduced") is not True:raise ValueError("Frozen trial selection/control changed")
    table=pd.read_csv(root/"trials.csv",float_precision="round_trip")
    if len(table)!=30 or table.number.tolist()!=list(range(30)):raise ValueError("Trial CSV coverage differs")
    for row,t in zip(table.to_dict("records"),trials):
        if row["state"]!=t["state"]:raise ValueError("Trial CSV state differs")
        close(row["objective_auc"],t["value"],"trial CSV objective")
        for key,val in t["parameters"].items():close(row[key],val,"trial CSV parameter")
    decision=b.read(root/"decision.json");best=b.read(root/"best_trial.json")
    if decision.get("selected_trial")!=21 or decision.get("selected_parameters")!=PARAMETERS or decision.get("selected_is_control") is not False or decision.get("screening_passed") is not True or decision.get("automatic_promotion") is not False or best!=b.read(root/"trials/trial_0021/result.json"):raise ValueError("Selected source decision differs")
    close(decision["selected_mean_auc"],HYBRID_AUC,"selected hybrid");close(winner["value"],HYBRID_AUC,"winning objective")
    verify_result(c,root,0,b.read(root/"trials/trial_0000/result.json"));summary=verify_result(c,root,21,best)
    if decision["comparisons"]!=best["comparisons"]:raise ValueError("Decision comparisons differ")
    primary=next(v for v in best["comparisons"] if v["candidate"]=="candidate_hybrid" and v["reference"]=="original_hybrid")
    if primary["mean_difference"]<=TOL or primary["wins"]<6:raise ValueError("Selected screening did not pass")
    tm,th=artifacts(trialroot,"prediction_manifest.json",{"submission.csv","component_predictions.csv","source_decision.json","source_best_trial.json","ensemble.json"},"ready_for_manual_submission");merge(hashes,th)
    if tm.get("candidate")!="elasticnet_optuna_trial_25" or tm.get("selected_trial")!=25 or tm.get("parameters")!=TRIAL25_PARAMETERS or tm.get("fits_performed")!=5 or tm.get("saved_and_reloaded_models_verified") is not True or tm.get("rows")!=cfg["expected_rows"] or tm.get("kaggle_upload_performed") is not False:raise ValueError("Wrong/unverified Trial25 export")
    merge(hashes,tm["source_input_and_code_sha256"])
    trainpath=str((ROOT/c["config"]["train_path"]).resolve())
    if tm["source_input_and_code_sha256"].get(trainpath)!=c["hashes"].get(trainpath) or trainpath not in c["hashes"]:raise ValueError("Trial25 does not bind the same training file")
    ts=Path(tm["source_study"]).resolve()
    if str(ts/"decision.json") not in c["hashes"] or str(ts/"best_trial.json") not in c["hashes"]:raise ValueError("Trial25 export not bound to same development source")
    if b.read(trialroot/"source_decision.json")!=b.read(ts/"decision.json") or b.read(trialroot/"source_best_trial.json")!=b.read(ts/"best_trial.json"):raise ValueError("Trial25 source snapshots differ")
    ensemble=b.read(trialroot/"ensemble.json")
    if ensemble.get("elasticnet_parameters")!=TRIAL25_PARAMETERS or ensemble.get("feature_order")!=list(c["x"].columns):raise ValueError("Trial25 evaluated feature/parameter recipe differs")
    if tm.get("submission_sha256")!=b.digest(trialroot/"submission.csv"):raise ValueError("Trial25 submission hash differs")
    inputs={str((ROOT/cfg[k]).resolve()) for k in ["test_path","sample_submission_path"]}
    if any(Path(p).resolve().is_relative_to(dest) for p in inputs):raise ValueError("Output overlaps prediction inputs")
    # Do not hash/read competition test or template in dry-run, even through a source map.
    if inputs.intersection(str(Path(p).resolve()) for p in hashes):raise ValueError("Development source map unexpectedly includes competition prediction inputs")
    for p in inputs:
        if not Path(p).is_file():raise FileNotFoundError(p)
    from src.submission import generative_trial21_export_v1
    code=[cfgpath,Path(__file__).resolve(),Path(inspect.getfile(generative_trial21_export_v1)),ROOT/"scripts/export_generative_trial21_v1.py"]
    merge(hashes,{str(p.resolve()):b.digest(p) for p in code});b.check(hashes)
    if dest.is_relative_to(ts) or ts.is_relative_to(dest):raise ValueError("Output overlaps original Trial25 study")
    c.update(output=dest,export_config=cfg,hashes=hashes,selected_parameters=best["parameters"],source_best=best,source_decision=decision,
        source_summary=summary,trial25_root=trialroot,trial25_manifest=tm,study_root=root)
    return c

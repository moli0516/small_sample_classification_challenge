"""Read-only snapshot of local v1 geometry, Trial25 and fixed hybrid evidence."""
import inspect
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from src.experiments import generative_followup_v1 as frozen

ROOT=Path(__file__).resolve().parents[2]
GMM_AUC=.9296401907271473
HYBRID_AUC=.9382748290356986
TOL=1e-12
DESIGN=dict(n_trials=30,sampler_seed=42,n_startup_trials=10,selected_features=[10,20,30,50],
    latent_dimensions=[4,6,8,12,16],n_components=[4,8,12,16],reg_covar=[1e-5,1e-1],label_smoothing=[1e-5,1e-2])


def merge_hashes(target,more):
    for p,h in more.items():
        if p in target and target[p]!=h:raise ValueError("Conflicting source hashes: "+p)
        target[p]=h


def prepare(config_path,output=None):
    b=frozen.base;p=(ROOT/config_path).resolve();cfg=b.read(p)
    if set(cfg)!={"followup_config","blend_study","output_dir","design"} or cfg["design"]!=DESIGN:
        raise ValueError("Frozen generative Optuna v1 design required")
    dest=(ROOT/(output or cfg["output_dir"])).resolve();source=(ROOT/cfg["blend_study"]).resolve()
    if dest.exists():raise FileExistsError("Fresh output required; no overwrite/resume")
    if dest.is_relative_to(source) or source.is_relative_to(dest):raise ValueError("Output overlaps source")
    # blend preparation verifies labelled data and original Trial25/K8 sources; it never opens competition test.
    c=frozen.prepare(str((ROOT/cfg["followup_config"]).resolve()),"blend",output=str(dest))
    hashes=dict(c["hashes"]);merge_hashes(hashes,frozen.verify_artifacts(source))
    m=b.read(source/"run_manifest.json")
    if m.get("action")!="blend" or m.get("competition_test_opened") is not False or m.get("completed_candidate_fits")!=0 or m.get("planned_candidate_fits")!=0:
        raise ValueError("Completed zero-fit/no-test blending study required")
    settings=b.read(source/"settings.json");decision=b.read(source/"decision.json")
    if settings["config"]!=c["followup_config"] or settings["source_design"]!=b.DESIGN or Path(settings["source_study"]).resolve()!=Path(c["source_study"]).resolve():
        raise ValueError("Source configuration/lineage changed")
    b.check(settings["input_and_code_sha256"]);merge_hashes(hashes,settings["input_and_code_sha256"])
    for package,expected in settings["versions"].items():
        if version(package)!=expected:raise ValueError("Source package changed: "+package)
    frame=pd.read_csv(source/"cohort.csv",dtype={"id":"string"},float_precision="round_trip")
    expected=c["frame"].copy();expected["id"]=expected.id.astype("string")
    if not frame.equals(expected) or b.read(source/"split_indices.json")!=c["groups"]:raise ValueError("Source cohort/split mismatch")
    if len(c["groups"])!=50 or len(frame)!=2500 or c["x"].shape!=(250,150):raise ValueError("Original250x150/50-fold experiment required")
    with np.load(source/"predictions.npz",allow_pickle=False) as z:
        names=["A_k8_tied","reference_trial25",*[f"blend_gmm{w:g}" for w in [.25,.5,.75]]]
        arrays={n:np.asarray(z[n],dtype=float).copy() for n in names}
    for v in arrays.values():b.probability(v,len(frame))
    if not np.allclose(arrays["A_k8_tied"],c["control"],rtol=0,atol=TOL) or not np.allclose(arrays["reference_trial25"],c["reference"],rtol=0,atol=TOL):raise ValueError("Frozen source predictions differ")
    for w in [.25,.5,.75]:
        if not np.allclose(arrays[f"blend_gmm{w:g}"],w*c["control"]+(1-w)*c["reference"],rtol=0,atol=TOL):raise ValueError("Source hybrid formula differs")
    _,repeats,summary=b.evaluate(c,arrays);means=summary.set_index("candidate").mean_auc
    if max([f"blend_gmm{w:g}" for w in [.25,.5,.75]],key=lambda n:means[n])!="blend_gmm0.5":raise ValueError("Source winner changed")
    if decision.get("selected_candidate")!="blend_gmm0.5" or decision.get("selected_spec")!={"gmm_weight":.5,"trial25_weight":.5} or decision.get("screening_against_both_controls") is not True:
        raise ValueError("Expected screening-passed equal-weight hybrid")
    for a,v in [(float(means["A_k8_tied"]),GMM_AUC),(float(means["blend_gmm0.5"]),HYBRID_AUC),(decision["selected_mean_auc"],HYBRID_AUC)]:
        if not np.isclose(a,v,rtol=0,atol=TOL):raise ValueError("Frozen score failed reproduction")
    for ref in ["A_k8_tied","reference_trial25"]:
        actual=b.compare(repeats,"blend_gmm0.5",ref)
        saved=[s for s in decision["comparisons"] if s["candidate"]=="blend_gmm0.5" and s["reference"]==ref]
        if len(saved)!=1 or actual["mean_difference"]<=TOL or actual["wins"]<6:raise ValueError("Source screening failed")
        for k in ["mean_difference","difference_sd","wins","losses","ties"]:
            if not np.isclose(actual[k],saved[0][k],rtol=0,atol=TOL):raise ValueError("Recorded source comparison differs")
    from src.features import generative_search_geometry_v1
    from src.experiments import generative_optuna_v1
    code=[p,Path(__file__).resolve(),Path(inspect.getfile(generative_search_geometry_v1)),Path(inspect.getfile(generative_optuna_v1)),ROOT/"scripts/run_generative_optuna_v1.py"]
    merge_hashes(hashes,{str(q.resolve()):b.digest(q) for q in code})
    if any(Path(q).resolve().is_relative_to(dest) for q in hashes):raise ValueError("Output overlaps watched input")
    b.check(hashes)
    c.update(output=dest,hashes=hashes,optuna_config=cfg,hybrid_control=arrays["blend_gmm0.5"],source_reproduction=summary)
    return c

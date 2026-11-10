"""Synthetic tests; real-data provenance preflight must run locally."""
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from src.experiments import rf_auto_search as m


class Helper:
    @staticmethod
    def _path(path): return Path(path).resolve()
    @staticmethod
    def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @staticmethod
    def _write(path, value): Path(path).write_text(json.dumps(value, allow_nan=False))


def seal(root):
    paths = [p for p in root.iterdir() if p.name != "run_manifest.json"]
    Helper._write(root / "run_manifest.json", dict(status="completed", completed_forest_fits=200,
        baseline_oof_verified=True, test_opened=False,
        artifact_sha256={p.name: Helper._sha(p) for p in paths}))


@pytest.fixture
def reference(tmp_path):
    rng = np.random.default_rng(9)
    x = pd.DataFrame(rng.normal(size=(250,150)))
    y = pd.Series([0,1] * 125)
    ids = pd.Series([f"{i:04d}" for i in range(250)], dtype="string")
    splits, rows, preds, baseline_rows = [], [], [], []
    for repeat in range(10):
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=repeat).split(x,y)):
            key = dict(repeat=repeat, split_seed=repeat, fold=fold)
            splits.append(dict(key, train_indices=tr.tolist(), valid_indices=va.tolist()))
            base = 1/(1+np.exp(-(.4*(2*y.iloc[va].to_numpy()-1)+x.iloc[va,0].to_numpy())))
            rf = .2 + .6*y.iloc[va].to_numpy()
            mixed = .9*base+.1*rf
            aucs = {m.BASELINE:roc_auc_score(y.iloc[va],base)}
            baseline_rows.extend(dict(key,row_index=int(i),l1_xgb_lgbm=float(base[j])) for j,i in enumerate(va))
            for alias in ("d3_leaf5","d3_leaf10","d5_leaf5","d5_leaf10"):
                aucs["rf_"+alias] = roc_auc_score(y.iloc[va],rf)
                aucs["blend_"+alias] = roc_auc_score(y.iloc[va],mixed)
                for j,i in enumerate(va):
                    preds.append(dict(key,configuration=alias,row_index=int(i),id=str(ids.iloc[i]),target=int(y.iloc[i]),
                        baseline=float(base[j]),rf=float(rf[j]),blend=float(mixed[j])))
            rows.extend(dict(key,candidate=n,auc=float(v)) for n,v in aucs.items())
    root=tmp_path/"reference";root.mkdir()
    sentinel=tmp_path/"input.txt";sentinel.write_text("original")
    hashes={str(sentinel):Helper._sha(sentinel)}
    source=dict(helper=Helper(),context=dict(x=x,y=y,ids=ids,splits=splits),config={"old":"fixed"},versions={"test":"synthetic"},
        hashes=hashes,oof=pd.DataFrame(baseline_rows).set_index(m.KEYS))
    Helper._write(root/"settings.json",dict(input_and_code_sha256=hashes,versions=source["versions"],config=source["config"],
        fixed_rf=dict(n_estimators=500,max_features="sqrt",bootstrap=True,random_state=42,n_jobs=1),
        grid=[dict(max_depth=d,min_samples_leaf=l) for d,l in [(3,5),(3,10),(5,5),(5,10)]],baseline_weight=.9,rf_weight=.1))
    Helper._write(root/"decision.json",dict(selected="blend_d3_leaf5",promote=True))
    folds=pd.DataFrame(rows);folds.to_csv(root/"fold_scores.csv",index=False)
    pd.DataFrame(preds).to_csv(root/"validation_probabilities.csv",index=False)
    table=folds.groupby(["candidate","repeat"]).auc.mean().groupby("candidate").mean().rename("mean_auc").reset_index()
    table.to_csv(root/"comparison.csv",index=False)
    seal(root)
    return root,source


def test_reference_accepted_and_no_deprecated_index_keyword(reference,monkeypatch):
    root,source=reference
    original=pd.DataFrame.set_index
    def checked(self,*a,**kw):
        if "verify_integrity" in kw: warnings.warn("deprecated index keyword",FutureWarning)
        return original(self,*a,**kw)
    monkeypatch.setattr(pd.DataFrame,"set_index",checked)
    control=m.load_reference(source,root)
    assert len(control["control_rows"])==150 and len(control["predictions"])==2500


@pytest.mark.parametrize("mutation",["hash","versions","weight","decision","id","target","rf","blend","auc","duplicate"])
def test_reference_tampering_rejected(reference,mutation):
    root,source=reference
    if mutation=="hash":
        (root/"settings.json").write_text("{}");
    elif mutation in ("versions","weight"):
        s=m.read(root/"settings.json")
        if mutation=="versions":s["versions"]={"wrong":"version"}
        else:s["rf_weight"]=.25
        Helper._write(root/"settings.json",s);seal(root)
    elif mutation=="decision":
        Helper._write(root/"decision.json",dict(selected="blend_d5_leaf5",promote=True));seal(root)
    elif mutation=="auc":
        s=pd.read_csv(root/"fold_scores.csv");s.loc[0,"auc"]+=.01;s.to_csv(root/"fold_scores.csv",index=False);seal(root)
    else:
        p=pd.read_csv(root/"validation_probabilities.csv",dtype={"id":"string"})
        if mutation=="id":p.loc[0,"id"]="wrong"
        if mutation=="target":p.loc[0,"target"]=1-int(p.loc[0,"target"])
        if mutation=="rf":p.loc[0,"rf"]=np.nan
        if mutation=="blend":p.loc[0,"blend"]=.123
        if mutation=="duplicate":p=pd.concat([p.iloc[[0]],p.iloc[:-1]],ignore_index=True)
        p.to_csv(root/"validation_probabilities.csv",index=False);seal(root)
    with pytest.raises((ValueError,KeyError)):m.load_reference(source,root)


def results(deltas=None):
    rows=[]
    for repeat in range(10):
        base=.8+.001*repeat
        for fold in range(5):
            key=dict(repeat=repeat,split_seed=repeat,fold=fold)
            rows.append(dict(key,candidate=m.BASELINE,auc=base))
            for tag,_ in m.FEATURES:
                delta=(deltas or {}).get(tag,[0]*10)[repeat]
                rows.append(dict(key,candidate="rf_"+tag,auc=.7))
                rows.append(dict(key,candidate="blend_"+tag,auc=base+delta))
    return pd.DataFrame(rows)


def test_exact_tie_keeps_control():
    _,_,d=m.summarize(results(),.0001)
    assert d["best_candidate"]=="blend_sqrt" and not d["promote_new_configuration"]


def test_practical_gain_gate():
    _,_,d=m.summarize(results({"fraction020":[.00003]*10}),.0001)
    assert d["best_candidate"]=="blend_fraction020" and not d["promote_new_configuration"]
    _,_,d=m.summarize(results({"fraction020":[.001]*10}),.0001)
    assert d["promote_new_configuration"] and d["selected_new_candidate"]=="blend_fraction020"


def test_no_fallback_when_best_has_only_five_wins():
    _,_,d=m.summarize(results({"fraction020":[.04]*5+[-.01]*5,"fraction050":[.001]*10}),.0001)
    assert d["best_candidate"]=="blend_fraction020" and d["selected_new_candidate"] is None


def test_incomplete_and_invalid_auc():
    with pytest.raises(ValueError):m.summarize(results().iloc[:-1],.0001)
    f=results();f.loc[0,"auc"]=np.nan
    with pytest.raises(ValueError):m.summarize(f,.0001)


@pytest.mark.parametrize("gain",[-1,True,float("nan"),.5])
def test_invalid_config(gain):
    with pytest.raises(ValueError):m.validate(dict(v1_config="a",reference_v1_run="b",output_dir="c",minimum_mean_gain=gain))


def prepared(reference,tmp_path):
    root,source=reference
    control=m.load_reference(source,root)
    return dict(helper=Helper(),context=source["context"],config=dict(minimum_mean_gain=.0001),
        output=tmp_path/"new_run",versions=source["versions"],hashes=source["hashes"],reference=control,source=source)


def test_mocked_hundred_fit_lifecycle(reference,tmp_path,monkeypatch,capsys):
    c=prepared(reference,tmp_path);calls=[]
    class FakeRF:
        def __init__(self,**kw):
            assert kw["max_features"] in (.2,.5) and kw["n_estimators"]==500 and kw["n_jobs"]==1
            self.classes_=np.array([0,1])
        def fit(self,x,y):
            assert len(x)==200
            self.training=set(x.index);calls.append(1)
        def predict_proba(self,x):
            assert len(x)==50 and self.training.isdisjoint(x.index)
            p=1/(1+np.exp(-x.iloc[:,2].to_numpy()))
            return np.column_stack([1-p,p])
    monkeypatch.setattr(m,"RandomForestClassifier",FakeRF)
    runner=m.RFAutoSearch();monkeypatch.setattr(runner,"_prepare",lambda *a:c)
    assert runner.run("unused",dry_run=True)["new_forest_fits"]==100
    assert not calls and not c["output"].exists()
    runner.run("unused")
    assert len(calls)==100 and "[100/100]" in capsys.readouterr().out
    manifest=m.read(c["output"]/"run_manifest.json")
    assert manifest["status"]=="completed" and manifest["completed_forest_fits"]==100
    assert not manifest["test_opened"] and not manifest["kaggle_upload_performed"]
    assert len(pd.read_csv(c["output"]/"fold_scores.csv"))==350
    assert len(pd.read_csv(c["output"]/"validation_probabilities.csv"))==7500
    for name,digest in manifest["artifact_sha256"].items(): assert Helper._sha(c["output"]/name)==digest
    with pytest.raises(FileExistsError):runner.run("unused")


def test_fit_warning_stops_search(reference,tmp_path,monkeypatch):
    c=prepared(reference,tmp_path)
    class BadRF:
        def __init__(self,**kw):pass
        def fit(self,x,y):
            warnings.warn("review fit",UserWarning)
            raise RuntimeError("failure")
    monkeypatch.setattr(m,"RandomForestClassifier",BadRF)
    runner=m.RFAutoSearch();monkeypatch.setattr(runner,"_prepare",lambda *a:c)
    with pytest.raises(RuntimeError):runner.run("unused")
    assert m.read(c["output"]/"run_manifest.json")["status"]=="failed_or_interrupted"
    assert m.read(c["output"]/"warnings.json")[0]["message"]=="review fit"
    assert not (c["output"]/"decision.json").exists()


def test_metadata_change_detected(tmp_path):
    p=tmp_path/"input";p.write_text("a")
    stamps=m.fingerprint([p]);p.write_text("changed")
    with pytest.raises(RuntimeError):m.check_metadata(stamps)


def test_real_rf_api_smoke():
    model=RandomForestClassifier(**dict(m.RF_FIXED,n_estimators=8),max_features=.2)
    x=np.arange(400).reshape(40,10);y=np.tile([0,1],20)
    model.fit(x,y)
    m.v1.probability(model.predict_proba(x)[:,1],40)

"""Actual sklearn elastic-net fits; source adapter mocked where historical data is absent."""
from pathlib import Path
from types import ModuleType
import sys
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from scripts import run_elasticnet_ensemble_v1 as m


def context(tmp_path):
    x,y=make_classification(n_samples=40,n_features=8,n_informative=4,random_state=5)
    x=pd.DataFrame(x,columns=[f"x{i}" for i in range(8)]);y=pd.Series(y)
    groups=[];rows=[];parts=[];offset=0;rng=np.random.default_rng(4)
    for r in range(10):
        for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
            groups.append(dict(repeat=r,fold=f,split_seed=r,train_indices=tr.tolist(),valid_indices=va.tolist(),start=offset,stop=offset+len(va)))
            offset+=len(va);parts.append(rng.uniform(.1,.9,(len(va),3)))
            rows.extend(dict(repeat=r,fold=f,row_index=int(i),id=f"000{i}",target=int(y.iloc[i])) for i in va)
    base=np.concatenate(parts);watch=tmp_path/"source.txt";watch.write_text("immutable")
    return dict(context=dict(x=x,y=y),groups=groups,frame=pd.DataFrame(rows),base=base,baseline=base.mean(axis=1),
                catboost_control=.9*base.mean(axis=1)+.1*rng.uniform(.1,.9,len(base)),elasticnet_hashes={str(watch):m.digest(watch)},
                output=tmp_path/"result",output_root=tmp_path/"suite",hashes={str(watch):m.digest(watch)})


def test_grid_and_actual_penalty():
    specs=m.model_settings();assert len(specs)==9
    for s in specs:
        model=m.build_model(s)
        classifier=model.named_steps["classifier"]
        assert classifier.solver=="saga" and classifier.l1_ratio==s["l1_ratio"]
        assert classifier.C==s["C"] and classifier.random_state==42
        assert classifier.max_iter==20000


def test_complete_actual_450_fit_run(tmp_path):
    c=context(tmp_path);result=m.execute(c)
    assert result["completed_fits"]==450
    root=c["output"]
    assert len(pd.read_csv(root/"model_summary.csv"))==38
    assert len(pd.read_csv(root/"comparison.csv"))==36
    assert len(pd.read_csv(root/"comparison_vs_catboost.csv"))==36
    assert len(pd.read_csv(root/"standalone_summary.csv"))==9
    assert len(__import__("json").loads((root/"coefficient_diagnostics.json").read_text()))==450
    decision=__import__("json").loads((root/"decision.json").read_text())
    assert decision["automatic_promotion"] is False
    manifest=__import__("json").loads((root/"run_manifest.json").read_text())
    for name,h in manifest["artifact_sha256"].items():assert m.digest(root/name)==h
    with pytest.raises(FileExistsError):m.execute(c)


def test_scaler_training_only(tmp_path):
    c=context(tmp_path);s=c["groups"][0];x=c["context"]["x"].copy()
    x.iloc[s["valid_indices"]]+=1000
    model=m.build_model(m.model_settings()[0]).fit(x.iloc[s["train_indices"]],c["context"]["y"].iloc[s["train_indices"]])
    assert np.allclose(model.named_steps["scaler"].mean_,x.iloc[s["train_indices"]].mean())
    assert not np.allclose(model.named_steps["scaler"].mean_,x.mean())


def test_exact_ensemble_formulas_no_mutation():
    b=np.array([[.1,.2,.3],[.7,.8,.9]]);before=b.copy();p=np.array([.4,.6]);baseline=b.mean(axis=1)
    preds,specs=m.compose(b,baseline,p,"en",dict(C=1,l1_ratio=.5))
    assert len(preds)==4 and len(specs)==4
    assert np.allclose(preds["en_replace_l1"],(p+b[:,1]+b[:,2])/3)
    assert np.allclose(preds["en_add_alpha0.1"],.9*baseline+.1*p)
    assert np.array_equal(before,b)


def test_convergence_warning_fails(tmp_path):
    c=context(tmp_path)
    class Bad:
        def fit(self,x,y):warnings.warn("test",ConvergenceWarning)
    with pytest.raises(RuntimeError,match="ConvergenceWarning"):
        m.execute(c,model_builder=lambda spec:Bad())
    manifest=__import__("json").loads((c["output"]/"run_manifest.json").read_text())
    assert manifest["status"]=="failed_or_interrupted" and not (c["output"]/"decision.json").exists()


def test_tie_counts(tmp_path):
    c=context(tmp_path)
    _,r,_=m.score(c,{"reference":c["baseline"],"same":c["baseline"]})
    _,stats=m.compare(r,["same"],"reference")
    assert stats.ties.iloc[0]==10 and stats.wins.iloc[0]==0


def test_invalid_probability():
    with pytest.raises(ValueError):m.probability([np.nan],1)


def test_hash_guard(tmp_path):
    p=tmp_path/"x";p.write_text("old");h={str(p):m.digest(p)};p.write_text("new")
    with pytest.raises(ValueError):m.check_hashes(h)


def test_source_adapter_no_retraining(tmp_path,monkeypatch):
    c=context(tmp_path);module=ModuleType("scripts.run_ensemble_research_suite_v1")
    module.prepare=lambda path:c
    _,_,ss=m.score(c,{"catboost_frozen":c["catboost_control"]})
    decision=dict(selected_candidate="cat_10_alpha0.1",selected_mean_auc=float(ss.mean_auc.iloc[0]))
    module.load_child=lambda c,task:(decision,{"prediction__cat_10_alpha0.1":c["catboost_control"]},{})
    monkeypatch.setitem(sys.modules,module.__name__,module)
    from scripts import __dict__ as script_globals
    monkeypatch.delitem(script_globals,"run_ensemble_research_suite_v1",raising=False)
    def forbidden(*args,**kwargs):raise AssertionError("Unexpected fit")
    monkeypatch.setattr(m.Pipeline,"fit",forbidden)
    prepared=m.prepare("unused",output=str(tmp_path/"new"))
    assert prepared["output"]==tmp_path/"new" and not prepared["output"].exists()

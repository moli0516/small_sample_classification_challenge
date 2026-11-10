import ast
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import pytest

# Test the supplied new source against a minimal suite contract without native libraries.
# When the real suite is installed, its normal module is used instead.
try:
    from scripts import run_ensemble_research_suite_v1 as suite
except ImportError:
    suite=ModuleType("scripts.run_ensemble_research_suite_v1")
    def probability(p,n):
        p=np.asarray(p,dtype=float)
        if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("Invalid probability")
        return p
    suite.probability=probability
    sys.modules[suite.__name__]=suite
from scripts import run_ensemble_exploratory_combine_v1 as m


def inputs():
    return np.array([[.1,.2,.3],[.7,.8,.9]]),np.array([.4,.6]),np.array([.5,.5]),np.array([.2,.8]),np.array([.3,.7])


def test_exact_formulas():
    b,cat,lgb,xgb,stack=inputs()
    control,p=m.compose(b,cat,.1,lgb,xgb,stack)
    assert np.allclose(control,.9*b.mean(axis=1)+.1*cat)
    assert np.allclose(p["cat_plus_lgbm_es"],.3*b[:,0]+.3*b[:,1]+.3*lgb+.1*cat)
    assert np.allclose(p["cat_plus_xgb_es"],.3*b[:,0]+.3*xgb+.3*b[:,2]+.1*cat)
    assert np.allclose(p["cat_plus_both_es"],.3*b[:,0]+.3*xgb+.3*lgb+.1*cat)
    assert np.allclose(p["cat_plus_stacking"],.9*stack+.1*cat)
    assert len(p)==4


def test_inputs_not_mutated():
    b,cat,lgb,xgb,stack=inputs();before=b.copy()
    m.compose(b,cat,.1,lgb,xgb,stack)
    assert np.array_equal(b,before)


def test_alpha_is_frozen():
    b,cat,lgb,xgb,stack=inputs()
    with pytest.raises(ValueError,match="fixes"):
        m.compose(b,cat,.25,lgb,xgb,stack)


def test_invalid_predictions_rejected():
    b,cat,lgb,xgb,stack=inputs();cat[0]=np.nan
    with pytest.raises(ValueError):m.compose(b,cat,.1,lgb,xgb,stack)


def test_no_training_calls_in_new_module():
    tree=ast.parse(Path(m.__file__).read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="fit" for n in ast.walk(tree))

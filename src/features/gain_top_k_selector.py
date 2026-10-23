"""Fold-local total-gain selection with deterministic ties and original column order."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.utils.validation import check_is_fitted


class GainTopKSelector(TransformerMixin,BaseEstimator):
    """Use fit() in a pipeline, or reuse a training-fitted booster via fit_from_booster()."""
    def __init__(self,k=10,estimator=None):
        self.k=k
        self.estimator=estimator

    def fit(self,X,y):
        if self.estimator is None:
            raise ValueError("fit requires an unfitted ranking estimator.")
        self.estimator_=clone(self.estimator)
        self.estimator_.fit(X,y)
        model=self.estimator_.named_steps["classifier"] if hasattr(self.estimator_,"named_steps") else self.estimator_
        return self.fit_from_booster(model.get_booster(),list(X.columns))

    def fit_from_booster(self,booster,feature_names):
        names=list(feature_names)
        if type(self.k) is not int or not 1<=self.k<=len(names):
            raise ValueError("k must be between 1 and the predictor count.")
        if not names or len(set(names))!=len(names) or any(not isinstance(n,str) for n in names):
            raise ValueError("Feature names must be unique strings.")
        scores=booster.get_score(importance_type="total_gain")
        if set(scores)-set(names):
            raise ValueError("Booster feature names differ from training predictors.")
        gains=np.array([scores.get(name,0.) for name in names],dtype=float)
        if not np.isfinite(gains).all() or (gains<0).any():
            raise ValueError("Invalid total gains.")
        self.feature_names_in_=np.asarray(names,dtype=object)
        self.n_features_in_=len(names)
        self.gains_=gains
        self.ranking_indices_=np.lexsort((np.arange(len(names)),-gains))
        self.support_=np.zeros(len(names),dtype=bool)
        self.support_[self.ranking_indices_[:self.k]]=True
        return self

    def transform(self,X):
        check_is_fitted(self,"support_")
        if not isinstance(X,pd.DataFrame) or list(X.columns)!=list(self.feature_names_in_):
            raise ValueError("Predictor names/order must match selector training input.")
        return X.loc[:,self.get_feature_names_out()].copy()

    def get_support(self,indices=False):
        check_is_fitted(self,"support_")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,"support_")
        if input_features is not None and list(input_features)!=list(self.feature_names_in_):
            raise ValueError("Input features differ from fitted selector.")
        return self.feature_names_in_[self.support_].copy()

    def ranked_features(self):
        check_is_fitted(self,"support_")
        return self.feature_names_in_[self.ranking_indices_[:self.k]].tolist()

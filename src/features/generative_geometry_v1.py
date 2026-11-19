"""Unsupervised geometry heuristic, fitted within the declared evaluation scope."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class GeometryProjector:
    def __init__(self,n_features=20,n_components=8):
        self.n_features=n_features;self.n_components=n_components

    def fit(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or not np.isfinite(x).all() or min(x.shape)<2:raise ValueError("Invalid geometry input")
        self.scaler_=StandardScaler().fit(x);z=self.scaler_.transform(x)
        variance=z.var(axis=0);good=variance>1e-12
        correlation=(z.T@z)/len(z);np.fill_diagonal(correlation,0)
        # Squared correlation energy plus a weak fourth-moment departure heuristic.
        # This does not identify true relevant features; all rankings are label-free.
        scores=(correlation**2).sum(axis=1)+.05*np.abs((z**4).mean(axis=0)-3)
        scores[~good]=-np.inf
        selected=np.argsort(-scores,kind="stable")[:min(self.n_features,int(good.sum()))]
        if len(selected)<1:raise ValueError("No nonconstant features")
        self.selected_indices_=np.sort(selected)
        rank=min(self.n_components,len(selected),len(z)-1)
        self.pca_=PCA(n_components=rank,svd_solver="full").fit(z[:,self.selected_indices_])
        self.diagnostics_=dict(selected_indices=self.selected_indices_.tolist(),feature_scores=[float(v) if np.isfinite(v) else None for v in scores],
            explained_variance_ratio=self.pca_.explained_variance_ratio_.tolist(),rows_fitted=len(x),components=rank)
        return self

    def transform(self,x):
        return self.pca_.transform(self.scaler_.transform(np.asarray(x,dtype=float))[:,self.selected_indices_])

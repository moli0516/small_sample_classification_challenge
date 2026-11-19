"""Plain standardized PCA control: no geometry-based feature selection."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PlainPCAProjector:
    def __init__(self,n_components=8):self.n_components=n_components
    def fit(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError("Invalid projector input")
        self.scaler_=StandardScaler().fit(x)
        self.pca_=PCA(n_components=min(self.n_components,x.shape[1],len(x)-1),svd_solver="full").fit(self.scaler_.transform(x))
        return self
    def transform(self,x):return self.pca_.transform(self.scaler_.transform(np.asarray(x,dtype=float)))

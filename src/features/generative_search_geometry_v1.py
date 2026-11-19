"""Search wrapper preserving the frozen v1 geometry arithmetic."""
from src.features.generative_geometry_v1 import GeometryProjector


class SearchGeometryProjector(GeometryProjector):
    def __init__(self,selected_features=20,latent_dimensions=8):
        self.requested_features=selected_features
        self.requested_dimensions=latent_dimensions
        super().__init__(n_features=selected_features,n_components=latent_dimensions)

    def fit(self,x):
        super().fit(x)
        self.search_diagnostics_=dict(requested_features=self.requested_features,
            effective_features=len(self.selected_indices_),requested_dimensions=self.requested_dimensions,
            effective_dimensions=int(self.pca_.n_components_),selected_indices=self.selected_indices_.tolist(),
            explained_variance_ratio=self.pca_.explained_variance_ratio_.tolist())
        return self

from _utils import Geom_t
from sklearn.metrics import pairwise_distances
import pandas as pd
import pytest

from anndata import AnnData


@pytest.fixture()
def adata_with_cost_matrix(adata_x: Geom_t, adata_y: Geom_t) -> AnnData:
    adata = adata_x.concatenate(adata_y, batch_key="batch")
    C = pairwise_distances(adata_x.obsm["X_pca"], adata_y.obsm["X_pca"]) ** 2
    adata.obs["batch"] = pd.to_numeric(adata.obs["batch"])
    adata.uns[0] = C / C.mean()  # TODO(@MUCDK) make a callback function and replace this part
    return adata

from typing import Tuple, Union, Optional

from scipy.sparse import csr_matrix
import scipy
import pandas as pd
import pytest

from jax.config import config

from anndata import AnnData
import scanpy as sc

config.update("jax_enable_x64", True)
from _utils import _get_random_trees
from sklearn.metrics import pairwise_distances

from jax import numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _utils import Geom_t, RTOL, ATOL


@pytest.fixture()
def adata_with_cost_matrix(adata_x: Geom_t, adata_y: Geom_t):
    adata = adata_x.concatenate(adata_y, batch_key="batch")
    C = pairwise_distances(adata_x.obsm["X_pca"], adata_y.obsm["X_pca"]) ** 2
    adata.obs["batch"] = pd.to_numeric(adata.obs["batch"])
    adata.uns[0] = C / C.mean()  # TODO(@MUCDK) make a callback function and replace this part
    return adata
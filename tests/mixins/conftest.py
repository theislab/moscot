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


_gt_temporal_adata = sc.read("tests/data/moscot_temporal_tests.h5ad")


@pytest.fixture()
def gt_temporal_adata() -> AnnData:
    return _gt_temporal_adata
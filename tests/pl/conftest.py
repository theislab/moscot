import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot._utils import _check_uns_keys
from moscot._constants._constants import AdataKeys, PlottingKeys, PlottingDefaults


@pytest.fixture
def adata_pl_cell_transition(gt_temporal_adata: AnnData) -> AnnData:
    _check_uns_keys(gt_temporal_adata, AdataKeys.UNS, PlottingKeys.CELL_TRANSITION, PlottingDefaults.CELL_TRANSITION)

    gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][PlottingDefaults.CELL_TRANSITION][
        "transition_matrix"
    ] = gt_temporal_adata.uns[f"cell_transition_10_105_forward"]
    gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][PlottingDefaults.CELL_TRANSITION][
        "source_annotation"
    ] = "cell_type"
    gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][PlottingDefaults.CELL_TRANSITION][
        "target_annotation"
    ] = "cell_type"
    gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][PlottingDefaults.CELL_TRANSITION][
        "source_key"
    ] = 0
    gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][PlottingDefaults.CELL_TRANSITION][
        "target_key"
    ] = 1

    return gt_temporal_adata


@pytest.fixture
def adata_pl_push(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    _check_uns_keys(adata_time, AdataKeys.UNS, PlottingKeys.PUSH, PlottingDefaults.PUSH)
    adata_time.uns[AdataKeys.UNS][PlottingKeys.PUSH][PlottingDefaults.PUSH]["temporal_key"] = "time"
    adata_time.obs[PlottingDefaults.PUSH] = np.abs(rng.randn(len(adata_time)))
    return adata_time


@pytest.fixture
def adata_pl_pull(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    _check_uns_keys(adata_time, AdataKeys.UNS, PlottingKeys.PULL, PlottingDefaults.PULL)
    adata_time.uns[AdataKeys.UNS][PlottingKeys.PULL][PlottingDefaults.PULL]["temporal_key"] = "time"
    adata_time.obs[PlottingDefaults.PULL] = np.abs(rng.randn(len(adata_time)))
    return adata_time


@pytest.fixture
def adata_pl_sankey(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    celltypes = ["A", "B", "C", "D", "E"]
    adata_time.obs["celltype"] = rng.choice(celltypes, size=len(adata_time))
    adata_time.obs["celltype"] = adata_time.obs["celltype"].astype("category")
    data1 = np.abs(rng.randn(5, 5))
    data2 = np.abs(rng.randn(5, 5))
    tm1 = pd.DataFrame(data=data1, index=celltypes, columns=celltypes)
    tm2 = pd.DataFrame(data=data2, index=celltypes, columns=celltypes)
    _check_uns_keys(adata_time, AdataKeys.UNS, PlottingKeys.SANKEY, PlottingDefaults.SANKEY)
    adata_time.uns[AdataKeys.UNS][PlottingKeys.SANKEY][PlottingDefaults.SANKEY]["transition_matrices"] = [tm1, tm2]
    adata_time.uns[AdataKeys.UNS][PlottingKeys.SANKEY][PlottingDefaults.SANKEY]["captions"] = ["0", "1"]
    adata_time.uns[AdataKeys.UNS][PlottingKeys.SANKEY][PlottingDefaults.SANKEY]["key"] = "celltype"

    return adata_time

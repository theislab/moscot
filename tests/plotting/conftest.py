import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot._constants._constants import Key, AdataKeys, PlottingKeys, PlottingDefaults


@pytest.fixture
def adata_pl_cell_transition(gt_temporal_adata: AnnData) -> AnnData:
    plot_vars = {
        "transition_matrix": gt_temporal_adata.uns[f"cell_transition_10_105_forward"],
        "source_annotation": "cell_type",
        "target_annotation": "cell_type",
        "source_key": 0,
        "target_key": 1,
    }
    Key.uns.set_plotting_vars(
        gt_temporal_adata, AdataKeys.UNS, PlottingKeys.CELL_TRANSITION, PlottingDefaults.CELL_TRANSITION, plot_vars
    )

    return gt_temporal_adata


@pytest.fixture
def adata_pl_push(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    plot_vars = {"temporal_key": "time"}
    Key.uns.set_plotting_vars(adata_time, AdataKeys.UNS, PlottingKeys.PUSH, PlottingDefaults.PUSH, plot_vars)
    adata_time.obs[PlottingDefaults.PUSH] = np.abs(rng.randn(len(adata_time)))
    return adata_time


@pytest.fixture
def adata_pl_pull(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    plot_vars = {"temporal_key": "time"}
    Key.uns.set_plotting_vars(adata_time, AdataKeys.UNS, PlottingKeys.PULL, PlottingDefaults.PULL, plot_vars)
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
    plot_vars = {"transition_matrices": [tm1, tm2], "captions": ["0", "1"], "key": "celltype"}
    Key.uns.set_plotting_vars(adata_time, AdataKeys.UNS, PlottingKeys.SANKEY, PlottingDefaults.SANKEY, plot_vars)

    return adata_time

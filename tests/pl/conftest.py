import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot._constants._constants import AdataKeys, PlottingKeys, PlottingDefaults


@pytest.fixture
def adata_pl_cell_transition(gt_temporal_adata: AnnData) -> AnnData:
    options = ["forward", "backward"]
    for i, key in enumerate([PlottingDefaults.CELL_TRANSITION, "cell_transition_backward"]):
        gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][key][
            "transition_matrix"
        ] = gt_temporal_adata.uns[f"cell_transition_10_105_{options[i]}"]
        gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][key]["source_annotation"] = "cell_type"
        gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][key]["target_annotation"] = "cell_type"
        gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][key]["source_key"] = 0
        gt_temporal_adata.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][key]["target_key"] = 1

    return gt_temporal_adata


@pytest.fixture
def adata_pl_push(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState
    adata_time.uns[AdataKeys.UNS][PlottingKeys.PUSH][PlottingDefaults.PUSH]["temporal_key"] = "time"
    adata_time.obs[PlottingDefaults.PUSH] = np.abs(rng.randn(len(adata_time)))
    return adata_time


@pytest.fixture
def adata_pl_pull(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState
    adata_time.uns[AdataKeys.UNS][PlottingKeys.PULL][PlottingDefaults.PULL]["temporal_key"] = "time"
    adata_time.obs[PlottingDefaults.PULL] = np.abs(rng.randn(len(adata_time)))
    return adata_time


@pytest.fixture
def adata_pl_sankey(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    celltypes = ["A", "B", "C", "D", "E"]
    adata_time.obs["celltype"] = rng.multinomial(len(adata_time), [0.1, 0.1, 0.2, 0.2, 0.3, 0.1])
    data1 = np.abs(rng.randn(5, 5))
    data2 = np.abs(rng.randn(5, 5))
    tm1 = pd.DataFrame(data=data1, index=celltypes, columns=celltypes)
    tm2 = pd.DataFrame(data=data2, index=celltypes, columns=celltypes)
    adata_time.uns[AdataKeys.UNS][PlottingKeys.SANKEY][PlottingDefaults.SANKEY]["transition_matrices"] = [tm1, tm2]
    adata_time.uns[AdataKeys.UNS][PlottingKeys.SANKEY][PlottingDefaults.SANKEY]["captions"] = ["0", "1"]
    adata_time.uns[AdataKeys.UNS][PlottingKeys.SANKEY][PlottingDefaults.SANKEY]["key"] = "time"

    return adata_time

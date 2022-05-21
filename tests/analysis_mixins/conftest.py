import pandas as pd
import pytest

import numpy as np

from anndata import AnnData


@pytest.fixture()
def adata_time_with_tmap(adata_time: AnnData) -> AnnData:
    adata = adata_time[adata_time.obs["time"].isin([0, 1])].copy()
    rng = np.random.RandomState(42)
    cell_types = ["cell_A", "cell_B", "cell_C", "cell_D"]

    cell_d1 = rng.multinomial(len(adata[adata.obs["time"] == 0]), [1 / len(cell_types)] * len(cell_types))
    cell_d2 = rng.multinomial(len(adata[adata.obs["time"] == 0]), [1 / len(cell_types)] * len(cell_types))
    a1 = np.concatenate(
        [["cell_A"] * cell_d1[0], ["cell_B"] * cell_d1[1], ["cell_C"] * cell_d1[2], ["cell_D"] * cell_d1[3]]
    ).flatten()
    a2 = np.concatenate(
        [["cell_A"] * cell_d2[0], ["cell_B"] * cell_d2[1], ["cell_C"] * cell_d2[2], ["cell_D"] * cell_d2[3]]
    ).flatten()

    adata.obs["cell_type"] = np.concatenate([a1, a2])
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    cell_numbers_source = dict(adata[adata.obs["time"] == 0].obs["cell_type"].value_counts())
    cell_numbers_target = dict(adata[adata.obs["time"] == 1].obs["cell_type"].value_counts())
    trans_matrix = np.abs(rng.randn(len(cell_types), len(cell_types)))
    trans_matrix = trans_matrix / trans_matrix.sum(axis=1, keepdims=1)

    cell_transition_gt = pd.DataFrame(data=trans_matrix, index=cell_types, columns=cell_types)

    blocks = []
    for cell_row in cell_types:
        block_row = []
        for cell_col in cell_types:
            sub_trans_matrix = np.abs(rng.randn(cell_numbers_source[cell_row], cell_numbers_target[cell_col]))
            sub_trans_matrix /= sub_trans_matrix.sum() * (1 / cell_transition_gt.loc[cell_row, cell_col])
            block_row.append(sub_trans_matrix)
        blocks.append(block_row)
    transport_matrix = np.block(blocks)
    adata.uns["transport_matrix"] = transport_matrix
    adata.uns["cell_transition_gt"] = cell_transition_gt
    return adata

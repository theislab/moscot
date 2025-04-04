import pytest

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

import anndata as ad
from anndata import AnnData

from tests._utils import Geom_t


@pytest.fixture
def adata_with_cost_matrix(adata_x: Geom_t, adata_y: Geom_t) -> AnnData:
    adata = ad.concat([adata_x, adata_y], label="batch", index_unique="-")
    C = pairwise_distances(adata_x.obsm["X_pca"], adata_y.obsm["X_pca"]) ** 2
    adata.obs["batch"] = pd.to_numeric(adata.obs["batch"])
    adata.uns[0] = C / C.mean()  # TODO(@MUCDK) make a callback function and replace this part
    return adata


@pytest.fixture
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


# keys for marginals
@pytest.fixture(
    params=[
        (None, None),
        ("left_marginals_balanced", "right_marginals_balanced"),
    ],
    ids=["default", "balanced"],
)
def marginal_keys(request):
    return request.param


sinkhorn_args_1 = {
    "epsilon": 0.7,
    "tau_a": 1.0,
    "tau_b": 1.0,
    "rank": 7,
    "initializer": "rank2",
    "initializer_kwargs": {},
    "jit": False,
    "threshold": 2e-3,
    "lse_mode": True,
    "norm_error": 2,
    "inner_iterations": 3,
    "min_iterations": 4,
    "max_iterations": 9,
    "gamma": 9.4,
    "gamma_rescale": False,
    "batch_size": None,  # in to_LRC() `batch_size` cannot be passed so we expect None.
    "scale_cost": "max_cost",
}


sinkhorn_args_2 = {  # no gamma/gamma_rescale as these are LR-specific
    "epsilon": 0.8,
    "tau_a": 0.9,
    "tau_b": 0.8,
    "rank": -1,
    "batch_size": 125,
    "initializer": "gaussian",
    "initializer_kwargs": {},
    "jit": True,
    "threshold": 3e-3,
    "lse_mode": False,
    "norm_error": 3,
    "inner_iterations": 4,
    "min_iterations": 1,
    "max_iterations": 2,
    "scale_cost": "mean",
}

linear_solver_kwargs1 = {
    "inner_iterations": 1,
    "min_iterations": 5,
    "max_iterations": 7,
    "lse_mode": False,
    "threshold": 5e-2,
    "norm_error": 4,
}

gw_args_1 = {  # no gamma/gamma_rescale/tolerances/ranks as these are LR-specific
    "epsilon": 0.5,
    "tau_a": 0.7,
    "tau_b": 0.8,
    "scale_cost": "max_cost",
    "rank": -1,
    "batch_size": 122,
    "initializer": None,
    "initializer_kwargs": {},
    "jit": True,
    "threshold": 3e-2,
    "min_iterations": 3,
    "max_iterations": 4,
    "gw_unbalanced_correction": True,
    "ranks": 4,
    "tolerances": 2e-2,
    "warm_start": False,
    "linear_solver_kwargs": linear_solver_kwargs1,
}

linear_solver_kwargs2 = {
    "inner_iterations": 3,
    "min_iterations": 7,
    "max_iterations": 8,
    "lse_mode": True,
    "threshold": 4e-2,
    "norm_error": 3,
    "gamma": 9.4,
    "gamma_rescale": False,
}

gw_args_2 = {
    "alpha": 0.4,
    "epsilon": 0.7,
    "tau_a": 1.0,
    "tau_b": 1.0,
    "scale_cost": "max_cost",
    "rank": 7,
    "batch_size": 123,
    "initializer": "rank2",
    "initializer_kwargs": {},
    "jit": False,
    "threshold": 2e-3,
    "min_iterations": 2,
    "max_iterations": 3,
    "gw_unbalanced_correction": False,
    "ranks": 3,
    "tolerances": 3e-2,
    # "linear_solver_kwargs": linear_solver_kwargs2,
}

gw_args_2 = {**gw_args_2, **linear_solver_kwargs2}

fgw_args_1 = gw_args_1.copy()
fgw_args_1["alpha"] = 0.6

fgw_args_2 = gw_args_2.copy()
fgw_args_2["alpha"] = 0.4

gw_solver_args = {
    "epsilon": "epsilon",
    "rank": "rank",
    "threshold": "threshold",
    "min_iterations": "min_iterations",
    "max_iterations": "max_iterations",
    "warm_start": "warm_start",
    "initializer": "initializer",
}

gw_lr_solver_args = {
    "epsilon": "epsilon",
    "rank": "rank",
    "threshold": "threshold",
    "min_iterations": "min_iterations",
    "max_iterations": "max_iterations",
    "initializer": "initializer",
}

gw_linear_solver_args = {
    "lse_mode": "lse_mode",
    "inner_iterations": "inner_iterations",
    "threshold": "threshold",
    "norm_error": "norm_error",
    "max_iterations": "max_iterations",
    "min_iterations": "min_iterations",
}

gw_lr_linear_solver_args = {
    "lse_mode": "lse_mode",
    "inner_iterations": "inner_iterations",
    "threshold": "threshold",
    "norm_error": "norm_error",
    "max_iterations": "max_iterations",
    "min_iterations": "min_iterations",
    "gamma": "gamma",
    "gamma_rescale": "gamma_rescale",
}

quad_prob_args = {
    "tau_a": "tau_a",
    "tau_b": "tau_b",
    "gw_unbalanced_correction": "gw_unbalanced_correction",
    "ranks": "ranks",
    "tolerances": "tolerances",
}

geometry_args = {"epsilon": "_epsilon_init", "scale_cost": "_scale_cost"}

pointcloud_args = {
    "batch_size": "_batch_size",
    "scale_cost": "_scale_cost",
}

lr_pointcloud_args = {
    "batch_size": "batch_size",
    "scale_cost": "_scale_cost",
}

sinkhorn_solver_args = {  # dictionary with key = moscot arg name, value = ott-jax attribute
    "lse_mode": "lse_mode",
    "threshold": "threshold",
    "norm_error": "norm_error",
    "inner_iterations": "inner_iterations",
    "min_iterations": "min_iterations",
    "max_iterations": "max_iterations",
    "initializer": "initializer",
    "initializer_kwargs": "initializer_kwargs",
}

lr_sinkhorn_solver_args = sinkhorn_solver_args.copy()
lr_sinkhorn_solver_args["gamma"] = "gamma"
lr_sinkhorn_solver_args["gamma_rescale"] = "gamma_rescale"

lin_prob_args = {
    "tau_a": "tau_a",
    "tau_b": "tau_b",
}

neurallin_cond_args_1 = {
    "batch_size": 8,
    "seed": 0,
    "iterations": 2,
    "valid_freq": 4,
}

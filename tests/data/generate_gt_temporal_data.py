from typing import Any, Dict, Tuple, TYPE_CHECKING
import sys

try:
    import wot  # please install WOT from commit hash`ca5e94f05699997b01cf5ae13383f9810f0613f6`"
except ImportError:
    raise ImportError(
        "Please install WOT from commit hash`ca5e94f05699997b01cf5ae13383f9810f0613f6`"
        + "with `pip install git+https://github.com/broadinstitute/wot.git@ca5e94f05699997b01cf5ae13383f9810f0613f6`"
    )

import os

from sklearn.metrics import pairwise_distances
import pandas as pd

import numpy as np

from anndata import AnnData
import scanpy as sc

from moscot._types import ArrayLike
from moscot.problems.time import TemporalProblem  # type: ignore[attr-defined]

eps = 0.5
lam1 = 1
lam2 = 10
key = "day"
key_1 = 10
key_2 = 10.5
key_3 = 11
local_pca = 50
tau_a = lam1 / (lam1 + eps)
tau_b = lam2 / (lam2 + eps)
seed = 42

config = {
    "eps": eps,
    "lam1": lam1,
    "lam2": lam2,
    "tau_a": tau_a,
    "tau_b": tau_b,
    "key": key,
    "key_1": key_1,
    "key_2": key_2,
    "key_3": key_3,
    "local_pca": local_pca,
    "seed": seed,
}


def _write_config(adata: AnnData) -> AnnData:
    adata.uns["eps"] = eps
    adata.uns["lam1"] = lam1
    adata.uns["lam2"] = lam2
    adata.uns["tau_a"] = tau_a
    adata.uns["tau_b"] = tau_b
    adata.uns["key"] = key
    adata.uns["key_1"] = key_1
    adata.uns["key_2"] = key_2
    adata.uns["key_3"] = key_3
    adata.uns["local_pca"] = local_pca
    adata.uns["seed"] = seed
    return adata


def _create_adata(data_path: str) -> AnnData:
    # follow instructions on https://broadinstitute.github.io/wot/ to download the data
    # icb path: /lustre/groups/ml01/workspace/moscot_paper/wot_data/data
    VAR_GENE_DS_PATH = os.path.join(data_path, "ExprMatrix.var.genes.h5ad")
    CELL_DAYS_PATH = os.path.join(data_path, "cell_days.txt")
    SERUM_CELL_IDS_PATH = os.path.join(data_path, "serum_cell_ids.txt")
    CELL_SETS_PATH = os.path.join(data_path, "major_cell_sets.gmt")

    adata = wot.io.read_dataset(VAR_GENE_DS_PATH, obs=[CELL_DAYS_PATH], obs_filter=SERUM_CELL_IDS_PATH)
    CELL_SETS_PATH = os.path.join(data_path, "major_cell_sets.gmt")
    cell_sets = wot.io.read_sets(CELL_SETS_PATH, as_dict=True)
    cell_to_type = {v[i]: k for k, v in cell_sets.items() for i in range(len(v))}
    df_cell_type = pd.DataFrame(cell_to_type.items(), columns=["0", "cell_type"]).set_index("0")
    adata.obs = pd.merge(adata.obs, df_cell_type, how="left", left_index=True, right_index=True)

    adata = adata[adata.obs["day"].isin([10, 10.5, 11])]
    adata.obs["cell_type"] = adata.obs["cell_type"].fillna("unknown")
    sc.pp.subsample(adata, n_obs=250, random_state=0)

    return adata


def _write_analysis_output(cdata: AnnData, tp2: TemporalProblem, config: Dict[str, Any]) -> AnnData:
    cdata.obs["cell_type"] = cdata.obs["cell_type"].astype("category")
    cdata.uns["cell_transition_10_105_backward"] = tp2.cell_transition(
        config["key_1"], config["key_2"], source_groups="cell_type", target_groups="cell_type", forward=False
    )
    cdata.uns["cell_transition_10_105_forward"] = tp2.cell_transition(
        config["key_1"], config["key_2"], source_groups="cell_type", target_groups="cell_type", forward=True
    )
    cdata.uns["interpolated_distance_10_105_11"] = tp2.compute_interpolated_distance(
        config["key_1"], config["key_2"], config["key_3"], seed=config["seed"]
    )
    cdata.uns["random_distance_10_105_11"] = tp2.compute_random_distance(
        config["key_1"], config["key_2"], config["key_3"], seed=config["seed"]
    )
    cdata.uns["time_point_distances_10_105_11"] = list(
        tp2.compute_time_point_distances(config["key_1"], config["key_2"], config["key_3"])
    )
    cdata.uns["batch_distances_10"] = tp2.compute_batch_distances(config["key_1"], "batch")
    return cdata


def _prepare(adata: AnnData, config: Dict[str, Any]) -> Tuple[AnnData, ArrayLike, ArrayLike, ArrayLike]:
    adata_12 = adata[adata.obs[config["key"]].isin([config["key_1"], config["key_2"]])].copy()
    adata_23 = adata[adata.obs[config["key"]].isin([config["key_2"], config["key_3"]])].copy()
    adata_13 = adata[adata.obs[config["key"]].isin([config["key_1"], config["key_3"]])].copy()

    sc.tl.pca(adata_12, n_comps=config["local_pca"])
    sc.tl.pca(adata_23, n_comps=config["local_pca"])
    sc.tl.pca(adata_13, n_comps=config["local_pca"])

    C_12 = pairwise_distances(
        adata_12[adata_12.obs[config["key"]] == config["key_1"]].obsm["X_pca"],
        adata_12[adata_12.obs[config["key"]] == config["key_2"]].obsm["X_pca"],
        metric="sqeuclidean",
    )
    C_12 /= C_12.mean()
    C_23 = pairwise_distances(
        adata_23[adata_23.obs[config["key"]] == config["key_2"]].obsm["X_pca"],
        adata_23[adata_23.obs[config["key"]] == config["key_3"]].obsm["X_pca"],
        metric="sqeuclidean",
    )
    C_23 /= C_23.mean()
    C_13 = pairwise_distances(
        adata_13[adata_13.obs[config["key"]] == config["key_1"]].obsm["X_pca"],
        adata_13[adata_13.obs[config["key"]] == config["key_3"]].obsm["X_pca"],
        metric="sqeuclidean",
    )
    C_13 /= C_13.mean()

    return (
        adata_12[adata_12.obs[config["key"]] == config["key_1"]].concatenate(
            adata_12[adata_12.obs[config["key"]] == config["key_2"]],
            adata_13[adata_13.obs[config["key"]] == config["key_3"]],
        ),
        C_12,
        C_23,
        C_13,
    )


def generate_gt_temporal_data(data_path: str) -> None:
    """Generate `gt_temporal_data` for tests."""
    adata = _create_adata(data_path)

    cdata, C_12, C_23, C_13 = _prepare(adata, config)
    n_1, n_2, n_3 = len(C_12), len(C_23), C_23.shape[1]
    cdata.obsp["cost_matrices"] = np.block(
        [
            [np.zeros((n_1, n_1)), C_12, C_13],
            [np.zeros((n_2, n_1)), np.zeros((n_2, n_2)), C_23],
            [np.zeros((n_3, n_1)), np.zeros((n_3, n_2)), np.zeros((n_3, n_3))],
        ]
    )

    if TYPE_CHECKING:
        assert isinstance(config["seed"], int)
    rng = np.random.RandomState(config["seed"])
    cdata.obs["batch"] = rng.choice((0, 1, 2), len(cdata))

    ot_model = wot.ot.OTModel(
        cdata, day_field="day", epsilon=config["eps"], lambda1=config["lam1"], lambda2=config["lam2"], local_pca=0
    )
    tmap_wot_10_105 = ot_model.compute_transport_map(config["key_1"], config["key_2"], cost_matrix=C_12).X
    tmap_wot_105_11 = ot_model.compute_transport_map(config["key_2"], config["key_3"], cost_matrix=C_23).X
    tmap_wot_10_11 = ot_model.compute_transport_map(config["key_1"], config["key_3"], cost_matrix=C_13).X

    tp = TemporalProblem(cdata)
    tp = tp.prepare(
        "day",
        callback="cost-matrix",
        subset=[(10, 10.5), (10.5, 11), (10, 11)],
        policy="explicit",
        callback_kwargs={"key": "cost_matrices"},
    )
    tp = tp.solve(epsilon=config["eps"], tau_a=config["tau_a"], tau_b=config["tau_b"])

    assert (tp[config["key_1"], config["key_2"]].xy.data_src == C_12).all()
    assert (tp[config["key_2"], config["key_3"]].xy.data_src == C_23).all()
    assert (tp[config["key_1"], config["key_3"]].xy.data_src == C_13).all()

    np.testing.assert_array_almost_equal(
        np.corrcoef(
            np.array(tp[config["key_1"], config["key_2"]].solution.transport_matrix).flatten(),
            tmap_wot_10_105.flatten(),
        ),
        1,
    )
    np.testing.assert_array_almost_equal(
        np.corrcoef(
            np.array(tp[config["key_2"], config["key_3"]].solution.transport_matrix).flatten(),
            tmap_wot_105_11.flatten(),
        ),
        1,
    )
    np.testing.assert_array_almost_equal(
        np.corrcoef(
            np.array(tp[config["key_1"], config["key_3"]].solution.transport_matrix).flatten(), tmap_wot_10_11.flatten()
        ),
        1,
    )

    cdata.uns["tmap_10_105"] = np.array(tp[config["key_1"], config["key_2"]].solution.transport_matrix)
    cdata.uns["tmap_105_11"] = np.array(tp[config["key_2"], config["key_3"]].solution.transport_matrix)
    cdata.uns["tmap_10_11"] = np.array(tp[config["key_1"], config["key_3"]].solution.transport_matrix)

    tp2 = TemporalProblem(cdata)
    tp2 = tp2.prepare(
        "day",
        subset=[(10, 10.5), (10.5, 11), (10, 11)],
        policy="explicit",
        callback_kwargs={"n_comps": 50},
    )
    tp2 = tp2.solve(epsilon=config["eps"], tau_a=config["tau_a"], tau_b=config["tau_b"], scale_cost="mean")

    np.testing.assert_array_almost_equal(
        np.array(tp[config["key_1"], config["key_2"]].solution.transport_matrix),
        np.array(tp2[config["key_1"], config["key_2"]].solution.transport_matrix),
    )
    np.testing.assert_array_almost_equal(
        np.array(tp[config["key_2"], config["key_3"]].solution.transport_matrix),
        np.array(tp2[config["key_2"], config["key_3"]].solution.transport_matrix),
    )
    np.testing.assert_array_almost_equal(
        np.array(tp[config["key_1"], config["key_3"]].solution.transport_matrix),
        np.array(tp2[config["key_1"], config["key_3"]].solution.transport_matrix),
    )

    cdata = _write_analysis_output(cdata, tp2, config)
    cdata = _write_config(cdata)

    cdata.write("tests/data/moscot_temporal_tests.h5ad")


if __name__ == "__main__":
    generate_gt_temporal_data(sys.argv[1])

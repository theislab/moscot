from sys import excepthook


try:
    import wot # please install WOT from commit hash`ca5e94f05699997b01cf5ae13383f9810f0613f6`"
except:
    ImportError("Please install WOT from commit hash`ca5e94f05699997b01cf5ae13383f9810f0613f6`")

import moscot
from anndata import AnnData
import numpy as np
import pandas as pd
from moscot.problems.time import TemporalProblem

import scanpy as sc
import scipy
import os
from sklearn.metrics import pairwise_distances

def generate_gt_temporal_data() -> None:
    data_path = "/home/icb/dominik.klein/data/wot"
    output_dir = "/home/icb/dominik.klein/data"

    # follow instructions on https://broadinstitute.github.io/wot/ to download the data
    VAR_GENE_DS_PATH = os.path.join(data_path, 'ExprMatrix.var.genes.h5ad')
    CELL_DAYS_PATH = os.path.join(data_path, 'cell_days.txt')
    SERUM_CELL_IDS_PATH = os.path.join(data_path, 'serum_cell_ids.txt')
    CELL_SETS_PATH = os.path.join(data_path, 'major_cell_sets.gmt')

    adata = wot.io.read_dataset(VAR_GENE_DS_PATH, obs=[CELL_DAYS_PATH], obs_filter=SERUM_CELL_IDS_PATH)
    CELL_SETS_PATH = os.path.join(data_path, 'major_cell_sets.gmt')
    cell_sets = wot.io.read_sets(CELL_SETS_PATH, as_dict=True)
    cell_to_type = {v[i]: k for k, v in cell_sets.items() for i in range(len(v))}
    df_cell_type = pd.DataFrame(cell_to_type.items(), columns=["0", "cell_type"]).set_index("0")
    adata.obs = pd.merge(adata.obs, df_cell_type, how="left", left_index=True, right_index=True)

    adata = adata[adata.obs["day"].isin([10, 10.5, 11])]
    adata.obs["cell_type"] = adata.obs["cell_type"].fillna("unknown")
    sc.pp.subsample(adata, n_obs=1000, random_state=0)

    eps = 0.5
    lam1 = 1
    lam2 =10
    key = "day"
    key_1 = 10
    key_2 = 10.5
    key_3 = 11
    local_pca = 50
    tau_a = lam1/(lam1+eps)
    tau_b = lam2/(lam2+eps)

    cdata.uns["config_solution"]= {"eps": eps, 
                               "tau_a": tau_a, 
                               "tau_b": tau_b, 
                               "key": key, 
                               "key_1": key_1, 
                               "key_2": key_2,
                               "key_3": key_3,
                               "local_pca": local_pca}

    adata_1 = adata[adata.obs[key] == key_1].copy()
    adata_2 = adata[adata.obs[key] == key_2].copy()
    adata_3 = adata[adata.obs[key] == key_3].copy()
    sc.tl.pca(adata_1, n_comps=local_pca)
    sc.tl.pca(adata_2, n_comps=local_pca)
    sc.tl.pca(adata_3, n_comps=local_pca)
    C_12 = pairwise_distances(adata_1.obsm["X_pca"], adata_2.obsm["X_pca"], metric="sqeuclidean")
    C_12 /= C_12.mean()
    C_23 = pairwise_distances(adata_2.obsm["X_pca"], adata_3.obsm["X_pca"], metric="sqeuclidean")
    C_23 /= C_23.mean()
    C_13 = pairwise_distances(adata_1.obsm["X_pca"], adata_3.obsm["X_pca"], metric="sqeuclidean")
    C_13 /= C_13.mean()
    cdata = adata_1.concatenate(adata_2, adata_3)

    n_1 = np.sum(adata_1_indices)
    n_2 = np.sum(adata_2_indices)
    n_3 = np.sum(adata_3_indices)
    cdata.obsp["cost_matrices"] = np.block([[np.zeros((n_1,n_1)), C_12, C_13],[np.zeros((n_2, n_1)),np.zeros((n_2, n_2)), C_23], [np.zeros((n_3, n_1)), np.zeros((n_3, n_2)), np.zeros((n_3, n_3))]])

    rng = np.random.RandomState(42)
    cdata.obs["batch"] = rng.choice((0, 1, 2), len(cdata))

    ot_model = wot.ot.OTModel(cdata, day_field="day", epsilon = eps, lambda1 = lam1, lambda2 = lam2, local_pca=0)
    tmap_wot_10_105 = ot_model.compute_transport_map(key_1, key_2, cost_matrix=C_12).X
    tmap_wot_105_11 = ot_model.compute_transport_map(key_2, key_3, cost_matrix=C_23).X
    tmap_wot_10_11 = ot_model.compute_transport_map(key_1, key_3, cost_matrix=C_13).X

    tp = TemporalProblem(cdata)
    tp.prepare("day", callback="cost-matrix", subset=[(10,10.5), (10.5,11), (10,11)], policy="explicit",
            callback_kwargs = {"key": "cost_matrices"})
    tp.solve(epsilon=eps, tau_a=tau_a, tau_b=tau_b)

    assert (tp[10,10.5].xy.data == C_12).all()
    assert (tp[10,11].xy.data == C_13).all()
    assert (tp[10.5,11].xy.data == C_23).all()

    np.testing.assert_array_almost_equal(np.corrcoef(np.array(tp[10,10.5].solution.transport_matrix).flatten(), tmap_wot_10_105.flatten()), 1)
    np.testing.assert_array_almost_equal(np.corrcoef(np.array(tp[10.5,11].solution.transport_matrix).flatten(), tmap_wot_105_11.flatten()), 1)
    np.testing.assert_array_almost_equal(np.corrcoef(np.array(tp[10,11].solution.transport_matrix).flatten(), tmap_wot_10_11.flatten()), 1)

    cdata.uns["tmap_10_105"] = np.array(tp[10,10.5].solution.transport_matrix)
    cdata.uns["tmap_105_11"] = np.array(tp[10.5,11].solution.transport_matrix)
    cdata.uns["tmap_10_11"] = np.array(tp[10,11].solution.transport_matrix)

    tp2 = TemporalProblem(cdata)
    tp2.prepare("day", subset=[(10,10.5), (10.5,11), (10,11)], policy="explicit", callback_kwargs={"joint_space": False})
    tp2.solve(epsilon=eps, tau_a=tau_a, tau_b=tau_b, scale_cost="mean")

    np.testing.assert_array_almost_equal(np.corrcoef(np.array(tp[10,10.5].solution.transport_matrix.flatten()), np.array(tp2[10,10.5].solution.transport_matrix.flatten()))[0,1], 1)
    np.testing.assert_array_almost_equal(np.corrcoef(np.array(tp[10.5,11].solution.transport_matrix.flatten()), np.array(tp2[10.5,11].solution.transport_matrix.flatten()))[0,1], 1)
    np.testing.assert_array_almost_equal(np.corrcoef(np.array(tp[10,11].solution.transport_matrix.flatten()), np.array(tp2[10,11].solution.transport_matrix.flatten()))[0,1], 1)

    cdata.obs["cell_type"] = cdata.obs["cell_type"].astype("category")
    cdata.uns["cell_transition_10_105_backward"] = tp2.cell_transition(10,10.5, early_cells="cell_type", late_cells="cell_type", forward=False)
    cdata.uns["cell_transition_10_105_forward"] = tp2.cell_transition(10,10.5, early_cells="cell_type", late_cells="cell_type", forward=True)
    cdata.uns["interpolated_distance_10_105_11"] = tp2.compute_interpolated_distance(10,10.5,11, seed=42)
    cdata.uns["random_distance_10_105_11"] = tp2.compute_random_distance(10,10.5,11, seed=42)
    cdata.uns["time_point_distances_10_105_11"] = list(tp2.compute_time_point_distances(10, 10.5, 11))
    cdata.uns["batch_distances_10"] = tp2.compute_batch_distances(10, "batch")
    
    cdata.write_h5ad("moscot_temporal_tests.h5ad")

if __name__=="__main__":
    generate_gt_temporal_data()
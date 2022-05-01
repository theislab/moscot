"""To generate data install `pip install git+https://github.com/giovp/spatial-alignment.git`"""

from math import cos, sin
import sys

from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

import numpy as np

from scanpy import logging as logg
from anndata import AnnData
import scanpy as sc
import anndata as ad

sys.path.append("/home/icb/giovanni.palla/code/spatial-alignment")
sys.path.append("/home/icb/giovanni.palla/code/spatial-alignment/data")
from data.simulated.generate_twod_data import generate_twod_data  # noqa: E402


def generate_data(
    n_views: int = 1,
    n_outputs: int = 200,
    grid_size: int = 20,
    n_latent_gps: int = 10,
    kernel_lengthscale: int = 6,
    kernel_variance: int = 2,
    seed: int = 42,
) -> None:
    # create warped dataset with gpssa
    logg.warning("Create warped adata.")
    torch.manual_seed(seed)
    X, Y, _, view_idx = generate_twod_data(
        n_views=n_views,
        n_outputs=n_outputs,
        grid_size=grid_size,
        n_latent_gps=n_latent_gps,
        kernel_lengthscale=kernel_lengthscale,
        kernel_variance=kernel_variance,
    )
    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y)
    adata = AnnData(Y)

    adata.obsm["spatial"] = X
    batch = np.ones(adata.shape[0])
    batch[view_idx[0]] = 0
    adata.obs["batch"] = pd.Categorical(batch.astype(int).astype(str))
    adata.obs["idx"] = np.hstack([view_idx[0], view_idx[0]])

    adata.obs_names_make_unique()
    logg.warning("Save warped adata.")
    adata.write("./adata_spatial_warped.h5ad")
    # use same dataset to create rotated version
    logg.warning("Create rotated adata.")
    adata1 = adata[adata.obs["batch"] == "0"].copy()
    adata2 = adata[adata.obs["batch"] == "1"].copy()
    adata3 = adata[adata.obs["batch"] == "1"].copy()
    adata3.obs["batch"] = pd.Categorical(np.repeat("2", adata3.shape[0]))
    adata3.obsm["spatial"] = (adata3.obsm["spatial"] + 1) + 2
    adata3.obsm["spatial"][adata3.obsm["spatial"] > 10] = adata3.obsm["spatial"][adata3.obsm["spatial"] > 10] * 1.5

    for adata, angle in zip([adata1, adata2, adata3], [0, 25, 60]):
        theta = np.deg2rad(angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        adata.obsm["spatial"] = np.dot(adata.obsm["spatial"], rot)

    sc.pp.subsample(adata1, fraction=0.95, random_state=seed)
    sc.pp.subsample(adata2, fraction=0.9, random_state=seed)
    sc.pp.subsample(adata3, fraction=0.85, random_state=seed)
    adata = ad.concat([adata1, adata2, adata3], label="batch_key")
    adata.obs_names_make_unique()
    logg.warning("Save rotated adata.")
    adata.write("./adata_spatial_rotated.h5ad")
    return


if __name__ == "__main__":
    generate_data()

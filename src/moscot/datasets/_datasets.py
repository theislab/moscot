# this file was adapted from https://github.com/theislab/cellrank/blob/master/cellrank/datasets/_datasets.py
from types import MappingProxyType
from typing import Any, Tuple, Union, Literal, Mapping, Optional
import os

from scipy.sparse import csr_matrix
import pandas as pd

import numpy as np

from scanpy import read
from anndata import AnnData

from moscot.datasets._utils import _get_random_trees

__all__ = ["mosta", "hspc", "drosophila", "tedsim", "sim_align", "simulate_data"]
PathLike = Union[os.PathLike, str]

_datasets = MappingProxyType(
    {
        "TedSim": (
            "https://figshare.com/ndownloader/files/38031258",
            (16382, 500),
        ),
        "mosta": (
            "https://figshare.com/ndownloader/files/37953852",
            (54134, 2000),
        ),
        "hspc": (
            "https://figshare.com/ndownloader/files/37993503",
            (4000, 2000),
        ),
        "adata_dm_sc": (
            "https://figshare.com/ndownloader/files/37984938",
            (1297, 2000),
        ),
        "adata_dm_sp": (
            "https://figshare.com/ndownloader/files/37984935",
            (3039, 82),
        ),
        "sim_align": (
            "https://figshare.com/ndownloader/files/37984926",
            (1200, 500),
        ),
    }
)


def _load_dataset_from_url(
    fpath: PathLike,
    backup_url: str,
    expected_shape: Tuple[int, int],
    sparse: bool = True,
    cache: bool = True,
    **kwargs: Any,
) -> AnnData:
    fpath = str(fpath)
    if not fpath.endswith(".h5ad"):
        fpath += ".h5ad"
    fpath = os.path.expanduser(fpath)

    adata = read(filename=fpath, backup_url=backup_url, sparse=sparse, cache=cache, **kwargs)
    if adata.shape != expected_shape:
        raise ValueError(f"Expected `AnnData` object to have shape `{expected_shape}`, found `{adata.shape}`.")

    return adata


def mosta(
    path: PathLike = "~/.cache/moscot/mosta.h5ad",
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """Preprocessed and extracted data as provided in :cite:`chen:22`.

    Includes embryo sections `E9.5`, `E2S1`, `E10.5`, `E2S1`, `E11.5`, `E1S2`.

    The :attr:`anndata.AnnData.X` is based on reprocessing of the counts data using
    :func:`scanpy.pp.normalize_total` and :func:`scanpy.pp.log1p`.

    Parameters
    ----------
    path
        Path where to save the file.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(path, *_datasets["mosta"], **kwargs)


def hspc(
    path: PathLike = "~/.cache/moscot/hspc.h5ad",
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """Subsampled and processed data from the `NeurIPS Multimodal Single-Cell Integration Challenge \
    <https://www.kaggle.com/competitions/open-problems-multimodal/data>`_.

    4000 cells were randomly selected after filtering the multiome training data of the donor `31800`.
    Subsequently, the top 2000 highly variable genes were selected. Peaks appearing in less than 5%
    of the cells were filtered out, resulting in 11595 peaks.

    Parameters
    ----------
    path
        Path where to save the file.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(path, *_datasets["hspc"], **kwargs)


def drosophila(
    path: PathLike = "~/.cache/moscot/drosophila.h5ad",
    spatial: bool = False,
    **kwargs: Any,
) -> AnnData:
    """Embryo of Drosophila melanogaster described in :cite:`Li-spatial:22`.

    Minimal pre-processing was performed, such as gene and cell filtering, as well as normalization,
    see the `processing steps <https://github.com/theislab/moscot-framework_reproducibility>`_.

    Parameters
    ----------
    path
        Path where to save the file.
    spatial
        Whether to return the spatial or the scRNA-seq dataset.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    path, _ = os.path.splitext(path)
    if spatial:
        return _load_dataset_from_url(path + "_sp.h5ad", *_datasets["adata_dm_sp"], **kwargs)
    return _load_dataset_from_url(path + "_sc.h5ad", *_datasets["adata_dm_sc"], **kwargs)


def tedsim(
    path: PathLike = "~/.cache/moscot/tedsim.h5ad",
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """Dataset simulated with TedSim :cite:`pan:22`.

    The data was simulated with asymmetric division rate of `0.2` and intermediate state step size of `0.2`.
    :attr:`anndata.AnnData.X` was preprocessed using :func:`scanpy.pp.normalize_total` and :func:`scanpy.pp.log1p`.

    The annotated data object contains the following fields:

    - :attr:`anndata.AnnData.obsm` ``['barcodes']``: barcodes.
    - :attr:`anndata.AnnData.obsp` ``['barcodes_cost']``: pre-computed barcode distances.
    - :attr:`anndata.AnnData.uns` ``['tree']``: lineage tree in the Newick format.

    Parameters
    ----------
    path
        Path where to save the file.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(path, *_datasets["TedSim"], **kwargs)


def sim_align(
    path: PathLike = "~/.cache/moscot/sim_align.h5ad",
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """Spatial transcriptomics simulated dataset as described in :cite:`Jones-spatial:22`.

    Parameters
    ----------
    path
        Location where the file is saved to.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(path, *_datasets["sim_align"], **kwargs)


def simulate_data(
    n_distributions: int = 2,
    cells_per_distribution: int = 20,
    n_genes: int = 60,
    key: Literal["day", "batch"] = "batch",
    var: float = 1,
    obs_to_add: Mapping[str, Any] = MappingProxyType({"celltype": 3}),
    marginals: Optional[Tuple[str, str]] = None,
    seed: int = 0,
    quad_term: Optional[Literal["tree", "barcode", "spatial"]] = None,
    lin_cost_matrix: Optional[str] = None,
    quad_cost_matrix: Optional[str] = None,
    **kwargs: Any,
) -> AnnData:
    """
    Simulate data.

    This function is used to generate data, mainly for the purpose of
    demonstrating certain functionalities of :mod:`moscot`.

    Parameters
    ----------
    n_distributions
        Number of distributions defined by `key`.
    cells_per_distribution
        Number of cells per distribution.
    n_genes
        Number of genes per simulated cell.
    key
        Key to identify distribution allocation.
    var
        Variance of one cell distribution
    obs_to_add
        Dictionary of names to add to columns of :attr:`anndata.AnnData.obs`
        and number of different values for this column.
    marginals
        Column names of :attr:`anndata.AnnData.obs` where to save the randomly
        generated marginals. If `None`, no marginals are generated.
    seed
        Random seed.
    quad_term
        Literal indicating whether to add costs corresponding to a specific problem setting.
        If `None`, no quadratic cost element is generated.
    lin_cost_matrix
        Key where to save the linear cost matrix. If `None`, no linear cost matrix is generated.
    quad_cost_matrix
        Key where to save the quadratic cost matrices. If `None`, no quadratic cost matrix is generated.

    Returns
    -------
    :class:`anndata.AnnData`.
    """
    rng = np.random.RandomState(42)
    adatas = [
        AnnData(
            X=csr_matrix(
                rng.multivariate_normal(
                    mean=kwargs.pop("mean", np.arange(n_genes)),
                    cov=kwargs.pop("cov", var * np.diag(np.ones(n_genes))),
                    size=cells_per_distribution,
                )
            )
        )
        for _ in range(n_distributions)
    ]
    adata = adatas[0].concatenate(*adatas[1:], batch_key=key)
    if key == "day":
        adata.obs["day"] = pd.to_numeric(adata.obs["day"])
    for k, val in obs_to_add.items():
        adata.obs[k] = rng.choice(range(val), len(adata))
    if marginals:
        adata.obs[marginals[0]] = rng.uniform(1e-5, 1, len(adata))
        adata.obs[marginals[1]] = rng.uniform(1e-5, 1, len(adata))
    if quad_term == "tree":
        adata.uns["trees"] = {}
        for i in range(n_distributions):
            adata.uns["trees"][i] = _get_random_trees(
                n_leaves=cells_per_distribution,
                n_trees=1,
                n_initial_nodes=kwargs.pop("n_initial_nodes", 5),
                leaf_names=[adata[adata.obs[key] == i].obs_names],
                seed=seed,
            )[0]
    if quad_term == "barcode":
        raise NotImplementedError
    return adata

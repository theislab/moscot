# this file was adapted from https://github.com/theislab/cellrank/blob/master/cellrank/datasets/_datasets.py
from types import MappingProxyType
from typing import Any, Tuple, Union, Literal, Mapping, Optional
import os

from scipy.sparse import csr_matrix
import pandas as pd

import numpy as np

from scanpy import read
from anndata import AnnData

from moscot._docs._docs import d
from moscot.datasets._utils import _get_random_trees

# TODO(michalk8): expose all
__all__ = ["simulation", "mosta", "hspc", "drosophila_sc", "drosophila_sp", "sim_align", "simulate_data"]
PathLike = Union[os.PathLike, str]

_datasets = MappingProxyType(
    {
        "tedsim_1024": (
            "https://figshare.com/ndownloader/files/35786069",
            (1536, 500),
        ),
        "tedsim_15360": (
            "https://figshare.com/ndownloader/files/36556515",
            (15360, 500),
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


def _load_dataset_from_url(fpath: PathLike, backup_url: str, expected_shape: Tuple[int, int], **kwargs: Any) -> AnnData:
    fpath = str(fpath)
    if not fpath.endswith(".h5ad"):
        fpath += ".h5ad"
    kwargs.setdefault("sparse", True)
    kwargs.setdefault("cache", True)
    fpath = os.path.expanduser(f"~/.cache/moscot/{fpath}")
    print(fpath)

    adata = read(filename=fpath, backup_url=backup_url, **kwargs)

    if adata.shape != expected_shape:
        raise ValueError(f"Expected `AnnData` object to have shape `{expected_shape}`, found `{adata.shape}`.")

    return adata


@d.dedent
def simulation(
    path: PathLike = "datasets/simulated",
    size: Literal[1024, 15360] = 1024,
    **kwargs: Any,
) -> AnnData:
    """
    Dataset simulated with TedSim :cite:`pan:21` with parameters TODO.

    Parameters
    ----------
    path_prefix
        Location where file is saved to with the filename completed by the `size`.
    size
        Number of cells corresponding to the latter of the two time points.
    kwargs
        TODO.

    Returns
    -------
    %(adata)s
    """
    if size not in (1024, 15360):
        raise ValueError(f"Invalid size `{size}`, available values are: `{(1024, 15360)}`.")
    return _load_dataset_from_url(f"{path}_{size}", *_datasets[f"tedsim_{size}"], **kwargs)


@d.dedent
def mosta(
    path: PathLike = "datasets/mosta",
    **kwargs: Any,
) -> AnnData:
    """
    Preprocessed and extracted data as provided in :cite:`chen:22`.

    The anndata object includes embryo sections E9.5 E2S1, E10.5 E2S1, E11.5, E1S2. The :attr:`anndata.AnnData.X`
    entry is based on reprocessing of the counts data consisting of :meth:`scanpy.pp.normalize_total` and
    :meth:`scanpy.pp.log1p`.

    Parameters
    ----------
    path
        Location where the file is saved to.
    kwargs
        TODO.

    Returns
    -------
    %(adata)s
    """
    return _load_dataset_from_url(path, *_datasets["mosta"], **kwargs)


@d.dedent
def hspc(
    path: PathLike = "datasets/hspc",
    **kwargs: Any,
) -> AnnData:
    """
    Subsampled and processed data of the `NeurIPS Multimodal Single-Cell Integration Challenge \
    <https://www.kaggle.com/competitions/open-problems-multimodal/data>`.

    4000 cells were randomly selected after filtering the training data of the
    NeurIPS Multimodal Single-Cell Integration Challenge for multiome data and donor `31800`.
    Subsequently, the 2000 most highly variable genes were selected as well as all peaks
    appearing in less than 5% of the cells were filtered out, resulting in 11,595 peaks.

    Parameters
    ----------
    path
        Location where the file is saved to.
    kwargs
        TODO.

    Returns
    -------
    %(adata)s
    """
    return _load_dataset_from_url(path, *_datasets["hspc"], **kwargs)


@d.dedent
def drosophila_sc(
    path: PathLike = "datasets/adata_dm_sc",
    **kwargs: Any,
) -> AnnData:
    """
    Single-cell transcriptomics of embryo of drosophila melanogaster \
    as described in :cite:`Li-spatial:22`.

    Minimal pre-processing was performed, such as gene and cell filtering
    as well as normalization. Processing steps at
    https://github.com/theislab/moscot-framework_reproducibility.

    Parameters
    ----------
    path
        Location where the file is saved to.
    kwargs
        TODO.

    Returns
    -------
    %(adata)s
    """
    return _load_dataset_from_url(path, *_datasets["adata_dm_sc"], **kwargs)


@d.dedent
def drosophila_sp(
    path: PathLike = "datasets/adata_dm_sp",
    **kwargs: Any,
) -> AnnData:
    """
    Spatial transcriptomics of embryo of drosophila melanogaster \
    as described in :cite:`Li-spatial:22`.

    Minimal pre-processing was performed, such as gene and cell filtering
    as well as normalization. Processing steps at
    https://github.com/theislab/moscot-framework_reproducibility.

    Parameters
    ----------
    path
        Location where the file is saved to.
    kwargs
        TODO.

    Returns
    -------
    %(adata)s
    """
    return _load_dataset_from_url(path, *_datasets["adata_dm_sp"], **kwargs)


@d.dedent
def sim_align(
    path: PathLike = "datasets/sim_align",
    **kwargs: Any,
) -> AnnData:
    """
    Spatial transcriptomics dataset simulation described in :cite:`Jones-spatial:22`.

    Parameters
    ----------
    path
        Location where the file is saved to.
    kwargs
        TODO.

    Returns
    -------
    %(adata)s
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
    """TODO Simulate data."""
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
        pass  # TODO
    return adata

import os
import pickle
import urllib.request
from itertools import combinations
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import mudata as mu

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

import anndata as ad
from scanpy.readwrite import _check_datafile_present_and_download

from moscot._types import PathLike

__all__ = [
    "bone_marrow",
    "c_elegans",
    "drosophila",
    "hspc",
    "mosta",
    "pancreas_multiome",
    "sim_align",
    "simulate_data",
    "zebrafish",
]


def mosta(
    path: PathLike = "~/.cache/moscot/mosta.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:  # pragma: no cover
    """Preprocessed and extracted data as provided in :cite:`chen:22`.

    Includes embryo sections `E9.5`, `E2S1`, `E10.5`, `E2S1`, `E11.5`, `E1S2`.

    The :attr:`anndata.AnnData.X` is based on reprocessing of the counts data using
    :func:`scanpy.pp.normalize_total` and :func:`scanpy.pp.log1p`.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/40569779",
        expected_shape=(54134, 2000),
        force_download=force_download,
        **kwargs,
    )


def hspc(
    path: PathLike = "~/.cache/moscot/hspc.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:  # pragma: no cover
    """CD34+ hematopoietic stem and progenitor cells from 4 healthy human donors.

    From the `NeurIPS Multimodal Single-Cell Integration Challenge
    <https://www.kaggle.com/competitions/open-problems-multimodal/data>`_.

    4000 cells were randomly selected after filtering the multiome training data of the donor `31800`.
    Subsequently, the top 2000 highly variable genes were selected. Peaks appearing in less than 5%
    of the cells were filtered out, resulting in 11595 peaks.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    dataset = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/37993503",
        expected_shape=(4000, 2000),
        force_download=force_download,
        **kwargs,
    )
    dataset.obs["day"] = dataset.obs["day"].astype("category")  # better solution to this?

    return dataset


def drosophila(
    path: PathLike = "~/.cache/moscot/drosophila.h5ad",
    *,
    spatial: bool,
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Embryo of Drosophila melanogaster described in :cite:`Li-spatial:22`.

    Minimal pre-processing was performed, such as gene and cell filtering, as well as normalization.

    Parameters
    ----------
    path
        Path where to save the file.
    spatial
        Whether to return the spatial or the scRNA-seq dataset.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    path, _ = os.path.splitext(path)
    if spatial:
        return _load_dataset_from_url(
            path + "_sp.h5ad",
            file_type="h5ad",
            backup_url="https://figshare.com/ndownloader/files/37984935",
            expected_shape=(3039, 82),
            force_download=force_download,
            **kwargs,
        )

    return _load_dataset_from_url(
        path + "_sc.h5ad",
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/37984938",
        expected_shape=(1297, 2000),
        force_download=force_download,
        **kwargs,
    )


def c_elegans(
    path: PathLike = "~/.cache/moscot/c_elegans.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> Tuple[ad.AnnData, nx.DiGraph]:  # pragma: no cover
    """scRNA-seq time-series dataset of C.elegans embryogenesis :cite:`packer:19`.

    Contains raw counts of 46,151 cells with at least partial lineage information.
    In addition, this downloads the known C. elegans lineage tree.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object and the lineage tree.
    """
    adata = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/39943585",
        expected_shape=(46151, 20222),
        force_download=force_download,
        **kwargs,
    )

    with urllib.request.urlopen("https://figshare.com/ndownloader/files/43064842") as fin:
        tree = nx.read_gml(fin)

    return adata, tree


def zebrafish(
    path: PathLike = "~/.cache/moscot/zebrafish.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> Tuple[ad.AnnData, Dict[str, nx.DiGraph]]:
    """Lineage-traced scRNA-seq time-series dataset of Zebrafish heart regeneration :cite:`hu:22`.

    Contains gene expression vectors, LINNAEUS :cite:`spanjaard:18` reconstructed lineage trees,
    a low-dimensional embedding, and additional metadata.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object and the lineage trees.
    """
    adata = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/39951073",
        expected_shape=(44014, 31466),
        force_download=force_download,
        **kwargs,
    )
    # TODO(michalk8): also cache or store in AnnData ad Newick + reconstruct?
    with urllib.request.urlopen("https://figshare.com/ndownloader/files/39951076") as fin:
        trees = pickle.load(fin)

    return adata, trees


def bone_marrow(
    path: PathLike = "~/.cache/moscot/bone_marrow.h5ad",
    *,
    rna: bool,
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Multiome data of bone marrow measurements :cite:`luecken:21`.

    Contains processed counts of 6,224 cells. The RNA data was filtered to 2,000 top
    highly variable genes, the ATAC data was filtered to 8,000 top highly variable
    peaks.

    Parameters
    ----------
    path
        Path where to save the file.
    rna
        Return the RNA data If :obj:`True`, otherwise return ATAC data.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    path, _ = os.path.splitext(path)
    if rna:
        return _load_dataset_from_url(
            path + "_rna.h5ad",
            file_type="h5ad",
            backup_url="https://figshare.com/ndownloader/files/40195114",
            expected_shape=(6224, 2000),
            force_download=force_download,
            **kwargs,
        )
    return _load_dataset_from_url(
        path + "_atac.h5ad",
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/41013551",
        expected_shape=(6224, 8000),
        force_download=force_download,
        **kwargs,
    )


def pancreas_multiome(
    rna_only: bool,
    path: PathLike = "~/.cache/moscot/pancreas_multiome.h5mu",
    force_download: bool = True,
    **kwargs: Any,
) -> Union[mu.MuData, ad.AnnData]:  # pragma: no cover
    """Pancreatic endocrinogenesis dataset published with the moscot manuscript :cite:`klein:25`.

    The dataset contains paired scRNA-seq and scATAC-seq data of pancreatic cells at embryonic days 14.5, 15.5, and
    16.5. The data was preprocessed and filtered as described in the manuscript, the raw data and the full processed
    data are available via `GEO <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi>`_ accession code GSE275562.

    Parameters
    ----------
    rna_only
        Only load the RNA data, resulting in a smaller file.
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`anndata.io.read_h5ad` if `rna_only`, else for :func:`mudata.read`.

    Returns
    -------
        :class:`mudata.MuData` object with RNA and ATAC data if `rna_only`, else :class:`anndata.AnnData` with RNA only.
    """
    if rna_only:
        return _load_dataset_from_url(
            path,
            file_type="h5ad",
            backup_url="https://figshare.com/ndownloader/files/49725087",
            expected_shape=(22604, 20242),
            force_download=force_download,
            **kwargs,
        )
    return _load_dataset_from_url(
        path,
        file_type="h5mu",
        backup_url="https://figshare.com/ndownloader/files/48782332",
        expected_shape=(22604, 271918),
        force_download=force_download,
    )


def tedsim(
    path: PathLike = "~/.cache/moscot/tedsim.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:  # pragma: no cover
    """Dataset simulated with TedSim :cite:`pan:22`.

    Simulated scRNA-seq dataset of a differentiation trajectory. For each cell, the dataset includes a (raw counts)
    gene expression vector as well as a lineage barcodes. The data was simulated with asymmetric division rate of
    :math:`0.2`, intermediate state step size of :math:`0.2` and contains the following fields:

    - :attr:`obsm['barcodes'] <anndata.AnnData.obsm>` - barcodes.
    - :attr:`obsp['cost_matrices'] <anndata.AnnData.obsp>` - precomputed lineage cost matrices.
    - :attr:`uns['tree'] <anndata.AnnData.uns>` - lineage tree in the
      `Newick format <https://en.wikipedia.org/wiki/Newick_format>`_.
    - :attr:`uns['couplings' ] <anndata.AnnData.uns>` - coupling matrix based on the ground-truth lineage tree.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/40178644",
        expected_shape=(8448, 500),
        force_download=force_download,
        **kwargs,
    )


def sciplex(
    path: PathLike = "~/.cache/moscot/sciplex.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:  # pragma: no cover
    """Perturbation dataset published in :cite:`srivatsan:20`.

    Transcriptomes of A549, K562, and mCF7 cells exposed to 188 compounds.
    Data obtained from :cite:`scperturb:24`.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/43381398",
        expected_shape=(799317, 110984),
        force_download=force_download,
        **kwargs,
    )


def sim_align(
    path: PathLike = "~/.cache/moscot/sim_align.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:  # pragma: no cover
    """Spatial transcriptomics simulated dataset as described in :cite:`Jones-spatial:22`.

    Parameters
    ----------
    path
        Location where the file is saved to.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/37984926",
        expected_shape=(1200, 500),
        force_download=force_download,
        **kwargs,
    )


def simulate_data(
    n_distributions: int = 2,
    cells_per_distribution: int = 20,
    n_genes: int = 60,
    key: Literal["day", "batch"] = "batch",
    var: float = 1.0,
    obs_to_add: Mapping[str, Any] = MappingProxyType({"celltype": 3}),
    marginals: Optional[Tuple[str, str]] = None,
    seed: int = 0,
    quad_term: Optional[Literal["tree", "barcode", "spatial"]] = None,
    lin_cost_matrix: Optional[str] = None,
    quad_cost_matrix: Optional[str] = None,
    **kwargs: Any,
) -> ad.AnnData:
    """Simulate data.

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
        Key where to save the linear cost matrix. It is generated according to the pairwise policy.
        If `None`, no linear cost matrix is generated.
    quad_cost_matrix
        Key where to save the quadratic cost matrices. If `None`, no quadratic cost matrix is generated.

    Returns
    -------
    :class:`anndata.AnnData`.
    """
    rng = np.random.RandomState(seed)
    adatas = [
        ad.AnnData(
            X=rng.multivariate_normal(
                mean=kwargs.pop("mean", np.arange(n_genes)),
                cov=kwargs.pop("cov", var * np.diag(np.ones(n_genes))),
                size=cells_per_distribution,
            ),
        )
        for _ in range(n_distributions)
    ]
    adata = ad.concat(adatas, label=key, index_unique="-")
    if key == "day":
        adata.obs["day"] = pd.to_numeric(adata.obs["day"]).astype("category")
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
                # TODO(michalk8): fix the seed
                seed=seed,
            )[0]
    if quad_term == "spatial":
        dim = kwargs.pop("spatial_dim", 2)
        adata.obsm["spatial"] = rng.normal(size=(adata.n_obs, dim))
    if quad_term == "barcode":
        n_intBCs = kwargs.pop("n_intBCs", 20)
        barcode_dim = kwargs.pop("barcode_dim", 10)
        adata.obsm["barcode"] = rng.choice(n_intBCs, size=(adata.n_obs, barcode_dim))
    if lin_cost_matrix is not None:
        adata.uns[lin_cost_matrix] = {}
        for i, j in combinations(range(n_distributions), 2):
            adata.uns[lin_cost_matrix][(str(i), str(j))] = np.abs(
                rng.normal(size=(cells_per_distribution, cells_per_distribution))
            )
    if quad_cost_matrix is not None:
        quad_costs = [
            (np.abs(rng.normal(size=(cells_per_distribution, cells_per_distribution)))) for i in range(n_distributions)
        ]
        quad_cm = block_diag(*quad_costs)
        np.fill_diagonal(quad_cm, 0)
        adata.obsp[quad_cost_matrix] = quad_cm
    return adata


def _load_dataset_from_url(
    fpath: PathLike,
    file_type: Literal["h5ad", "h5mu"],
    *,
    backup_url: str,
    expected_shape: Tuple[int, int],
    force_download: bool = False,
    **kwargs: Any,
) -> Union[ad.AnnData, mu.MuData]:
    # TODO: make nicer once https://github.com/scverse/mudata/issues/76 resolved
    fpath = os.path.expanduser(fpath)
    assert file_type in ["h5ad", "h5mu"], f"Invalid type `{file_type}`. Must be one of `['h5ad', 'h5mu']`."
    if not fpath.endswith(file_type):
        fpath += f".{file_type}"
    if force_download and os.path.exists(fpath):
        os.remove(fpath)
    if not _check_datafile_present_and_download(backup_url=backup_url, path=fpath):
        raise FileNotFoundError(f"File `{fpath}` not found or download failed.")
    data = ad.read_h5ad(filename=fpath, **kwargs) if file_type == "h5ad" else mu.read_h5mu(filename=fpath, backed=False)

    if data.shape != expected_shape:
        data_str = "MuData" if file_type == "h5mu" else "AnnData"
        raise ValueError(f"Expected {data_str} object to have shape `{expected_shape}`, found `{data.shape}`.")

    return data


def _get_random_trees(
    n_leaves: int, n_trees: int, n_initial_nodes: int = 50, leaf_names: Optional[List[List[str]]] = None, seed: int = 42
) -> List[nx.DiGraph]:
    rng = np.random.RandomState(42)
    if leaf_names is not None:
        assert len(leaf_names) == n_trees
        for i in range(n_trees):
            assert len(leaf_names[i]) == n_leaves
    trees = []
    for tree_idx in range(n_trees):
        tempG = nx.random_labeled_tree(n_initial_nodes, seed=seed)
        G = nx.DiGraph()
        G.add_edges_from(tempG.edges)
        leaves = [x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1]
        inner_nodes = list(set(G.nodes()) - set(leaves))
        leaves_updated = leaves.copy()
        for i in range(n_leaves - len(leaves)):
            G.add_node(n_initial_nodes + i)
            G.add_edge(rng.choice(inner_nodes, 1)[0], n_initial_nodes + i)
            leaves_updated.append(n_initial_nodes + i)
        assert len(leaves_updated) == n_leaves
        if leaf_names is not None:
            relabel_dict = {leaves_updated[i]: leaf_names[tree_idx][i] for i in range(len(leaves_updated))}
            G = nx.relabel_nodes(G, relabel_dict)
        trees.append(G)

    return trees

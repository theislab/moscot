# this file was adapted from https://github.com/theislab/cellrank/blob/master/cellrank/datasets/_datasets.py
from typing import Any, Tuple, Union
import os

from scanpy import read
from anndata import AnnData

from moscot._docs import d

__all__ = ["simulation", "mosta", "hspc"]
PathLike = Union[os.PathLike, str]

_datasets = {
    "tedsim_1024": (
        "https://figshare.com/ndownloader/files/35786069",
        (1536, 500),
    ),
    "tedsim_15360": (
        "https://figshare.com/ndownloader/files/36556515",
        (15360, 500),
    ),
    "mosta": (
        "https://figshare.com/ndownloader/files/36703611",
        (54134, 2000),
    ),
    "hspc": (
        "https://figshare.com/ndownloader/files/36704517",
        (2000, 2000),
    ),
}


def _load_dataset_from_url(fpath: PathLike, backup_url: str, expected_shape: Tuple[int, int], **kwargs: Any) -> AnnData:
    fpath = str(fpath)
    if not fpath.endswith(".h5ad"):
        fpath += ".h5ad"
    kwargs.setdefault("sparse", True)
    kwargs.setdefault("cache", True)

    adata = read(filename=fpath, backup_url=backup_url, **kwargs)

    if adata.shape != expected_shape:
        raise ValueError(f"Expected `anndata.AnnData` object to have shape `{expected_shape}`, found `{adata.shape}`.")

    return adata


@d.dedent
def simulation(
    path_prefix: PathLike = "datasets/simulated_",
    size: int = 1024,
    **kwargs: Any,
) -> AnnData:
    """
    Dataset simulated with TedSim :cite:`Pan2021` with parameters TODO.

    Parameters
    ----------
    path_prefix
        Location where file is saved to with the filename completed by the `size`.
    size
        Number of cells corresponding to the latter of the two time points.

    Returns
    -------
    %(adata)s
    """
    _sizes = [1024, 15360]

    path = path_prefix + str(size)  # type: ignore[operator]
    if size not in _sizes:
        raise NotImplementedError(f"Available sizes are {_sizes}.")
    return _load_dataset_from_url(path, *_datasets[f"tedsim_{size}"], **kwargs)


@d.dedent
def mosta(
    path: PathLike = "datasets/mosta",
    **kwargs: Any,
) -> AnnData:
    """
    Preprocessed and extracted data as provided in :cite:`CHEN20221777`.

    The anndata object includes embryo sections E9.5 E2S1, E10.5 E2S1, E11.5, E1S2. The :attr:`anndata.AnnData.X`
    entry is based on reprocessing of the counts data consisting of :meth:`scanpy.pp.normalize_total` and
    :meth:`scanpy.pp.log1p`.

    Parameters
    ----------
    path
        Location where the file is saved to.

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

    2000 cells were randomly selected after filtering the training data of the
    NeurIPS Multimodal Single-Cell Integration Challenge for multiome data and donor `31800`.
    Subsequently, the 2000 most highly variable genes were selected as well as all peaks
    appearing in less than 5% of the cells were filtered out, resulting in 11,607 peaks.

    Parameters
    ----------
    path
        Location where the file is saved to.

    Returns
    -------
    %(adata)s
    """
    return _load_dataset_from_url(path, *_datasets["hspc"], **kwargs)

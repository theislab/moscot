# this file was adapted from https://github.com/theislab/cellrank/blob/master/cellrank/datasets/_datasets.py
from typing import Any, Tuple, Union
import os

from scanpy import read
from anndata import AnnData

from moscot._docs import d

__all__ = ["simulation"]
PathLike = Union[os.PathLike, str]

_datasets = {
    "tedsim_1024": (
        "https://figshare.com/ndownloader/files/35786069",
        (1536, 500),
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
    Dataset simulated with TedSim `cite:Pan2021` with parameters TODO.

    Parameters
    ----------
    %(dataset.parameters)s

    Returns
    -------
    %(adata)s
    """
    _sizes = [1024]

    path = path_prefix + str(size)  # type: ignore[operator]
    if size not in _sizes:
        raise NotImplementedError(f"Available sizes are {_sizes}.")
    return _load_dataset_from_url(path, *_datasets[f"tedsim_{size}"], **kwargs)

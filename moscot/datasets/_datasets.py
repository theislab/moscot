# this file was adapted from https://github.com/theislab/cellrank/blob/master/cellrank/datasets/_datasets.py
from typing import Any, Tuple, Union
from pathlib import Path
import os


from scanpy import read
from anndata import AnnData

from moscot._docs import d

__all__ = ("tedsim_1024",)


_datasets = {
    "tedsim_1024": (
        "https://figshare.com/ndownloader/files/35786069",
        (1536, 500),
    ),
}


def _load_dataset_from_url(
    fpath: Union[str, Path], url: str, expected_shape: Tuple[int, int], **kwargs: Any
) -> AnnData:
    fpath = str(fpath)
    if not fpath.endswith(".h5ad"):
        fpath += ".h5ad"

    # if os.path.isfile(fpath): TODO: logging
    #    logg.debug(f"Loading dataset from `{fpath!r}`") TODO: logging
    # else: TODO: logging
    #    logg.debug(f"Downloading dataset from `{url!r}` as `{fpath!r}`") TODO: logging

    dirname, _ = os.path.split(fpath)
    try:
        if not os.path.isdir(dirname):
            # logg.debug(f"Creating directory `{dirname!r}`") TODO: logging
            os.makedirs(dirname, exist_ok=True)
    except OSError:
        pass
        # logg.debug(f"Unable to create directory `{dirname!r}`. Reason `{e}`") TODO: logging

    kwargs.setdefault("sparse", True)
    kwargs.setdefault("cache", True)

    adata = read(fpath, backup_url=url, **kwargs)

    if adata.shape != expected_shape:
        raise ValueError(f"Expected `anndata.AnnData` object to have shape `{expected_shape}`, found `{adata.shape}`.")

    adata.var_names_make_unique()

    return adata


@d.dedent
def tedsim_1024(
    path: Union[str, Path] = "datasets/tedsim_512.h5ad",
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
    return _load_dataset_from_url(path, *_datasets["tedsim_1024"], **kwargs)

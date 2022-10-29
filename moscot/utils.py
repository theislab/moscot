from typing import List

from scipy.sparse import csr_matrix

from anndata import AnnData

from moscot._types import ArrayLike


def cost_to_obsp(
    adata: AnnData,
    key: str,
    cost_matrix: ArrayLike,
    obs_names_source: List[str],
    obs_names_target: List[str],
) -> AnnData:
    """
    Add cost_matrix to :attr:`adata.obsp`.

    Parameters
    ----------
    %(adata)s

    key
        Key of :attr:`anndata.AnnData.obsp` to inscribe or add the `cost_matrix` to.
    cost_matrix
        A cost matrix to be saved to the :class:`anndata.AnnData` instance.
    obs_names_source
        List of indices corresponding to the rows of the cost matrix
    obs_names_target
        List of indices corresponding to the columns of the cost matrix

    Returns
    -------
    :class:`anndata.AnnData` with modified/added :attr:`anndata.AnnData.obsp`.
    """
    obs_names = adata.obs_names
    if not set(obs_names_source).issubset(set(obs_names)):
        raise ValueError("TODO. `obs_names_source` is not a subset of `adata.obs_names`.")
    if not set(obs_names_target).issubset(set(obs_names)):
        raise ValueError("TODO. `obs_names_target` is not a subset of `adata.obs_names`.")
    obs_names_intersection = set(obs_names_source).intersection(set(obs_names_target))
    if len(obs_names_intersection) and set(obs_names_source) != set(obs_names_target):
        raise ValueError(
            "TODO. `obs_names_source` and `obs_names_target` must contain the same values or have empty intersection."
        )
    other_obs_names = set(obs_names) - set(obs_names_source) - set(obs_names_target)
    adata_reordered = adata[list(obs_names_source) + list(obs_names_target) + list(other_obs_names)]
    if key in adata.obsp:
        obsp_layer = adata_reordered.obsp[key]
    else:
        obsp_layer = csr_matrix((adata.n_obs, adata.n_obs))  # lil_matrix has no view in anndata.AnnData
    col_ilocs = (
        range(len(obs_names_target))
        if len(obs_names_intersection)
        else range(len(obs_names_source), len(obs_names_source) + len(obs_names_target))
    )
    obsp_layer[0 : len(obs_names_source), col_ilocs] = cost_matrix
    adata_reordered.obsp[key] = obsp_layer
    return adata_reordered[obs_names]

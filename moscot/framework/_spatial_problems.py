from __future__ import annotations

from typing import Any, Tuple, Optional, Sequence

import numpy as np

from anndata import AnnData

from moscot.framework.BaseProblem import BaseProblem
from moscot._constants._pkg_constants import Key


class BaseSpatialProblem(BaseProblem):
    """Base spatial problem."""

    def __init__(self, adata: AnnData, spatial_key: str = Key.obsm.spatial) -> None:
        """Init docs."""
        _assert_spatial_basis(adata, spatial_key)
        self._adata = adata
        self._rep = adata.obsm[spatial_key]
        super().__init__(adata=adata, rep=self._rep)

    def prepare(
        self,
        reference_key: str,
        target_key: str,
        batch_key: str = Key.obs.batch_key,
        alt_var: Optional[str | None] = None,
        **kwargs: Any,
    ) -> None:
        """Prepare docs."""
        _assert_value_in_obs(self.adata, batch_key, reference_key)
        _assert_value_in_obs(self.adata, batch_key, target_key)

        # subset adata.obs to get query and reference
        super().prepare(**kwargs)


class SpatialMappingProblem(BaseSpatialProblem):
    """Spatial Mapping Problem."""

    def prepare(
        self,
        reference_vars: Optional[Sequence[str] | str | None] = None,
        **kwargs: Any,
    ):
        """Prepare docs."""
        _assert_vars(self.adata, reference_vars)
        super().prepare(**kwargs)


class SpatialAlignmentProblem(BaseSpatialProblem):
    """Spatial Alignment Problem."""

    def prepare(
        self,
        **kwargs: Any,
    ):
        """Prepare docs."""
        super().prepare(**kwargs)

    def align(
        *,
        copy: bool = False,
    ) -> Tuple[Sequence[np.ndarray], Tuple[Sequence[float], Sequence[float]]] | None:
        """Align docs."""
        if copy:
            pass  # return aligned coordinates, rotation and translation factors


def _assert_spatial_basis(adata: AnnData, key: str) -> None:
    if key not in adata.obsm:
        raise KeyError(f"Spatial basis `{key}` not found in `adata.obsm`.")


def _assert_value_in_obs(adata: AnnData, key: str, val: Any) -> None:
    if key not in adata.obs:
        raise KeyError(f"Cluster key `{key}` not found in `adata.obs`.")
    if not isinstance(val, str):
        val = [val]
    if np.setdiff1d(val, adata.obs[key].unique()).shape[0] == 0:
        raise ValueError(f"`values: {val}` not found in `adata.obs[{key}]`.")


def _assert_vars(adata: AnnData, var: Sequence[str] | str) -> None:
    if isinstance(var, str):
        var = [var]
    if np.setdiff1d(var, adata.var_names).shape[0] == 0:
        raise ValueError(f"`var: {var}` not found in `adata.var_names`.")

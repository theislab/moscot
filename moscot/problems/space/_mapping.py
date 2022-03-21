from __future__ import annotations

from typing import Any, List, Mapping, Optional

from anndata import AnnData

from moscot.backends.ott import GWSolver, FGWSolver
from moscot.problems._base_problem import GeneralProblem
from moscot.mixins._spatial_analysis import SpatialMappingAnalysisMixin
from moscot.problems._compound_problem import SingleCompoundProblem


class MappingProblem(SingleCompoundProblem, SpatialMappingAnalysisMixin):
    """Mapping problem."""

    def __init__(
        self,
        adata_sc: AnnData,
        adata_sp: Optional[AnnData] = None,
        use_reference: bool = False,
        var_names: List[str] | None = None,
        solver_jit: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Init method."""
        self._adata_sp = adata_sp
        self._adata_sc = adata_sc
        self._use_reference = use_reference

        # filter genes
        self._filtered_vars = self._filter_vars(adata_sc, adata_sp, var_names, use_reference)
        solver = (
            FGWSolver(jit=solver_jit) if use_reference and self._filtered_vars is not None else GWSolver(jit=solver_jit)
        )
        self._adata_ref = adata_sc[:, self._filtered_vars] if self._filtered_vars is not None else adata_sc
        super().__init__(
            adata_sp[:, self._filtered_vars] if self._filtered_vars is not None else adata_sp, solver=solver, **kwargs
        )

    @property
    def adata_sp(self) -> AnnData:
        """Return spatial adata."""
        return self._adata_sp

    @property
    def adata_sc(self) -> AnnData:
        """Return single cell adata."""
        return self._adata_sc

    @property
    def filtered_vars(self) -> List[str]:
        """Return filtered variables."""
        return self._filtered_vars

    @property
    def _adata_tgt(self):
        """Return adata reference."""
        return self._adata_ref

    def prepare(
        self,
        attr_sc: str | Mapping[str, Any],
        attr_sp: str | Mapping[str, Any],
        attr_joint: str | Mapping[str, Any] | None = None,
        key: str | None = None,
        **kwargs: Any,
    ) -> GeneralProblem:
        """Prepare method."""
        attr_sc = {"attr": "obsm", "key": f"{attr_sc}"} if isinstance(attr_sc, str) else attr_sc
        attr_sp = {"attr": "obsm", "key": f"{attr_sp}"} if isinstance(attr_sp, str) is str else attr_sp
        attr_joint = {"x_attr": "X", "y_attr": "X"} if attr_joint is None else attr_joint

        if self._use_reference and self.filtered_vars is not None:
            return super().prepare(x=attr_sp, y=attr_sc, xy=attr_joint, policy="external_star", key=key, **kwargs)
        else:
            return super().prepare(x=attr_sp, y=attr_sc, policy="external_star", key=key, **kwargs)

    def solve(
        self,
        epsilon: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> GeneralProblem:
        """Solve method."""
        return super().solve(epsilon=epsilon, alpha=alpha, tau_a=tau_a, tau_b=tau_b, rank=rank, **kwargs)

    def _mask(self, key: Any, mask, adata: AnnData) -> AnnData:
        if key is self._policy._SENTINEL:
            return adata
        return super()._mask(key, mask, adata)

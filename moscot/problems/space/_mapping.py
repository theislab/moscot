from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from typing_extensions import Literal

import numpy.typing as npt

from anndata import AnnData

from moscot.backends.ott import GWSolver, FGWSolver
from moscot.problems._base_problem import GeneralProblem
from moscot.problems._subset_policy import Axis_t, DummyPolicy, SubsetPolicy, ExternalStarPolicy
from moscot.mixins._spatial_analysis import SpatialMappingAnalysisMixin
from moscot.problems._compound_problem import BaseProblem, SingleCompoundProblem


class MappingProblem(SingleCompoundProblem, SpatialMappingAnalysisMixin):
    """Mapping problem."""

    def __init__(
        self,
        adata_sp: AnnData,
        adata_sc: AnnData,
        var_names: Optional[Sequence[Any]] = None,
        solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ):
        """Init method."""
        super().__init__(adata_sp, **kwargs)
        self._adata_sc = adata_sc
        self._filtered_vars = self._filter_vars(var_names=var_names)

        self.solver = GWSolver(**solver_kwargs) if self._filtered_vars is None else FGWSolver(**solver_kwargs)

    def _create_policy(
        self,
        policy: Literal["external_star"] = "external_star",
        key: Optional[str] = None,
        axis: Axis_t = "obs",
        **_: Any,
    ) -> SubsetPolicy:
        if key is None:
            return DummyPolicy(self.adata, axis=axis)
        return ExternalStarPolicy(self.adata, key=key, axis=axis)

    def _create_problem(
        self,
        src: Any,
        src_mask: npt.ArrayLike,
        *_,
        **kwargs: Any,
    ) -> BaseProblem:
        adata_sp = self._mask(src_mask)
        return self._base_problem_type(
            adata_sp[:, self.filtered_vars] if self.filtered_vars is not None else adata_sp,
            self.adata_sc[:, self.filtered_vars] if self.filtered_vars is not None else self.adata_sc,
            solver=self._solver,
            **kwargs,
        )

    @property
    def adata_sc(self) -> AnnData:
        """Return single cell adata."""
        return self._adata_sc

    @property
    def filtered_vars(self) -> Optional[Sequence[Any]]:
        """Return filtered variables."""
        return self._filtered_vars

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

        if self.filtered_vars is None:
            return super().prepare(x=attr_sp, y=attr_sc, policy="external_star", key=key, **kwargs)
        return super().prepare(x=attr_sp, y=attr_sc, xy=attr_joint, policy="external_star", key=key, **kwargs)

from types import MappingProxyType
from typing import Any, Union, Mapping, Optional, Sequence

from typing_extensions import Literal

import numpy.typing as npt

from anndata import AnnData

from moscot.backends.ott import GWSolver, FGWSolver
from moscot.solvers._base_solver import ProblemKind
from moscot.problems._subset_policy import Axis_t, DummyPolicy, SubsetPolicy, ExternalStarPolicy
from moscot.mixins._spatial_analysis import SpatialMappingAnalysisMixin
from moscot.problems._compound_problem import GeneralProblem, SingleCompoundProblem


class MappingProblem(SingleCompoundProblem, SpatialMappingAnalysisMixin):
    """Mapping problem."""

    def __init__(
        self,
        adata_sc: AnnData,
        adata_spatial: AnnData,
        var_names: Optional[Sequence[Any]] = None,
        solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ):
        """Init method."""
        super().__init__(adata_spatial, **kwargs)
        self._adata_sc = adata_sc

        self.filtered_vars = var_names
        self.solver = GWSolver(**solver_kwargs) if self.filtered_vars is None else FGWSolver(**solver_kwargs)

    def _create_policy(
        self,
        policy: Literal["external_star"] = "external_star",
        key: Optional[str] = None,
        axis: Axis_t = "obs",
        **kwargs: Any,
    ) -> SubsetPolicy:
        if key is None:
            return DummyPolicy(self.adata, axis=axis, **kwargs)
        return ExternalStarPolicy(self.adata, key=key, axis=axis, **kwargs)

    def _create_problem(
        self,
        src: Any,
        tgt: Any,
        src_mask: npt.ArrayLike,
        tgt_mask: npt.ArrayLike,
        **kwargs: Any,
    ) -> GeneralProblem:
        adata_sp = self._mask(src_mask)
        return self._base_problem_type(
            adata_sp[:, self.filtered_vars] if self.filtered_vars is not None else adata_sp,
            self.adata_sc[:, self.filtered_vars] if self.filtered_vars is not None else self.adata_sc,
            solver=self._solver,
            **kwargs,
        )

    def prepare(
        self,
        sc_attr: Union[str, Mapping[str, Any]],
        spatial_key: str = "spatial",
        joint_attr: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        batch_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "MappingProblem":
        """Prepare method."""
        x = {"attr": "obsm", "key": spatial_key}
        y = {"attr": "obsm", "key": sc_attr} if isinstance(sc_attr, str) else sc_attr

        if self.filtered_vars is None and self.solver.problem_kind == ProblemKind.QUAD_FUSED:
            raise ValueError("TODO: wrong problem choice.")

        if joint_attr is None and self.solver.problem_kind == ProblemKind.QUAD_FUSED:
            kwargs["callback"] = "pca_local"

        return super().prepare(x=x, y=y, xy=joint_attr, policy="external_star", key=batch_key, **kwargs)

    @property
    def adata_sc(self) -> AnnData:
        """Return single cell adata."""
        return self._adata_sc

    @property
    def filtered_vars(self) -> Optional[Sequence[Any]]:
        """Return filtered variables."""
        return self._filtered_vars

    @filtered_vars.setter
    def filtered_vars(self, value: Optional[Sequence[str]]) -> None:
        """Return filtered variables."""
        self._filtered_vars = self._filter_vars(var_names=value)

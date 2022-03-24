from types import MappingProxyType
from typing import Any, Mapping, Optional

from typing_extensions import Literal

from anndata import AnnData

from moscot.backends.ott import FGWSolver
from moscot.solvers._base_solver import ProblemKind
from moscot.mixins._spatial_analysis import SpatialAlignmentAnalysisMixin
from moscot.problems._compound_problem import CompoundProblem


class AlignmentProblem(CompoundProblem, SpatialAlignmentAnalysisMixin):
    """Spatial alignment problem."""

    def __init__(
        self,
        adata: AnnData,
        solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ):
        """Init method."""
        super().__init__(adata, solver=FGWSolver(**solver_kwargs), **kwargs)
        self._spatial_key: Optional[str] = None

    def prepare(
        self,
        spatial_key: str = "spatial",
        attr_joint: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        policy: Literal["sequential", "star"] = "sequential",
        subset_key: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> "AlignmentProblem":
        """Prepare method."""
        if policy not in ("sequential", "star"):
            raise ValueError("TODO: return error message")
        self._spatial_key = spatial_key
        # TODO: check for spatial key
        x = {"attr": "obsm", "key": self.spatial_key}
        y = {"attr": "obsm", "key": self.spatial_key}

        if attr_joint is None and self.solver.problem_kind == ProblemKind.QUAD_FUSED:
            kwargs["callback"] = "pca_local"

        return super().prepare(x=x, y=y, xy=attr_joint, policy=policy, key=subset_key, reference=reference, **kwargs)

    @property
    def spatial_key(self) -> Optional[str]:
        """Return problems."""
        return self._spatial_key

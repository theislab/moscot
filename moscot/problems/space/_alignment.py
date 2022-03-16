from __future__ import annotations

from typing import Any, Mapping, Optional

from typing_extensions import Literal

try:
    pass
except ImportError:
    pass

from anndata import AnnData

from moscot.backends.ott import FGWSolver
from moscot.problems._base_problem import GeneralProblem
from moscot.problems._compound_problem import SingleCompoundProblem


class SpatialAlignmentProblem(SingleCompoundProblem):
    """Spatial alignment problem."""

    def __init__(
        self,
        adata: AnnData,
        solver_jit: Optional[bool] = None,
    ):
        """Init method."""
        solver = FGWSolver(jit=solver_jit)
        super().__init__(adata, solver=solver)

    @property
    def adata(
        self,
    ) -> AnnData:
        """Return adata."""
        return self._adata

    def prepare(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit", "external_star"] = "sequential",
        spatial_key: str = "spatial",
        attr_joint: Mapping[str, Any] = None,
        rank: int = None,
        **kwargs: Any,
    ) -> GeneralProblem:
        """Prepare method."""
        x = {"attr": "obsm", "key": f"{spatial_key}"}
        y = {"attr": "obsm", "key": f"{spatial_key}"}
        attr_joint = {"x_attr": "X", "y_attr": "X"} if attr_joint is None else attr_joint

        super().prepare(x=x, y=y, xy=attr_joint, policy=policy, **kwargs)

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

from __future__ import annotations

from typing import Any, Mapping, Optional

try:
    pass
except ImportError:
    pass

from typing import Optional


from anndata import AnnData

from moscot.backends.ott import FGWSolver
from moscot.problems._base_problem import GeneralProblem
from moscot.problems._compound_problem import SingleCompoundProblem


class SpatialAlignmentProblem(SingleCompoundProblem):
    def __init__(
        self,
        adata: AnnData,
        rank: Optional[int] = None,
        solver_jit: Optional[bool] = None,
    ):

        solver = FGWSolver(rank=rank, jit=solver_jit)
        super().__init__(adata, solver=solver)

    @property
    def adata(
        self,
    ) -> AnnData:
        return self._adata

    def prepare(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit", "external_star"] = "sequential",
        spatial_key: str = "spatial",
        attr_joint: Optional[Mapping[str, Any]] = {"x_attr": "X", "y_attr": "X"},
        **kwargs: Any,
    ) -> GeneralProblem:

        x = {"attr": "obsm", "key": f"{spatial_key}"}
        y = {"attr": "obsm", "key": f"{spatial_key}"}

        super().prepare(x=x, y=y, xy=attr_joint, policy=policy, **kwargs)

    def solve(
        self,
        epsilon: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        **kwargs: Any,
    ) -> GeneralProblem:

        return super().solve(epsilon=epsilon, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs)

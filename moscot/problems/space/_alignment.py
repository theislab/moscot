from __future__ import annotations

from typing import Any, Mapping, Optional

from typing_extensions import Literal

from anndata import AnnData

from moscot.backends.ott import FGWSolver
from moscot.problems._base_problem import GeneralProblem
from moscot.problems._compound_problem import CompoundProblem


class AlignmentProblem(CompoundProblem):
    """Spatial alignment problem."""

    def __init__(
        self,
        adata: AnnData,
        solver_jit: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Init method."""
        self._adata = adata
        solver = FGWSolver(jit=solver_jit)
        super().__init__(adata, solver=solver, **kwargs)

    @property
    def adata(self) -> AnnData:
        """Return adata."""
        return self._adata

    @property
    def problems(self) -> GeneralProblem:
        """Return problems."""
        return self._problems

    @property
    def spatial_key(self) -> str:
        """Return problems."""
        return self._spatial_key

    def prepare(
        self,
        spatial_key: str | Mapping[str, Any] = "spatial",
        attr_joint: Mapping[str, Any] = None,
        policy: Literal["sequential", "star"] = "sequential",
        key: str | None = None,
        reference: str | None = None,
        **kwargs: Any,
    ) -> GeneralProblem:
        """Prepare method."""
        # TODO: check for spatial key
        x = {"attr": "obsm", "key": f"{spatial_key}"}
        y = {"attr": "obsm", "key": f"{spatial_key}"}
        attr_joint = {"x_attr": "X", "y_attr": "X"} if attr_joint is None else attr_joint

        self._spatial_key = spatial_key

        return super().prepare(x=x, y=y, xy=attr_joint, policy=policy, key=key, reference=reference, **kwargs)

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

from typing import Any, Optional
import warnings

from ot import sinkhorn, sinkhorn_unbalanced

import numpy.typing as npt

from moscot.solvers._output import BaseSolverOutput
from moscot.backends.pot._output import POTOutput
from moscot.solvers._base_solver import ProblemKind, ContextlessBaseSolver
from moscot.solvers._tagged_array import TaggedArray

__all__ = ("SinkhornSolver", "GWSolver", "FGWSolver")


class SinkhornSolver(ContextlessBaseSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        epsilon: Optional[float] = None,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        a = [] if a is None else a
        b = [] if b is None else b
        epsilon = 1e-2 if epsilon is None else epsilon

        # TODO(michalk8): data
        # TODO(michalk8): tau_a in base solver none
        print(tau_a, "T")
        return a, b, x.data, epsilon, tau_a

    def _solve(self, data: Any, **kwargs: Any) -> BaseSolverOutput:
        # TODO(michalk8): default a/b based on backend
        kwargs["log"] = True  # TODO(michalk8): enable for loss?
        kwargs["warn"] = True
        a, b, M, epsilon, tau_a = data
        with warnings.catch_warnings(record=True) as record:
            if tau_a is None:
                T, log = sinkhorn(a=a, b=b, M=M, reg=epsilon, **kwargs)
            else:
                T, log = sinkhorn_unbalanced(a=a, b=b, M=M, reg=epsilon, reg_m=tau_a, **kwargs)
            converged = True  # TODO(michalk8)

        try:
            cost = log["err"][-1]
        except IndexError:
            cost = float("inf")

        return POTOutput(T, cost=cost, converged=converged)

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.LINEAR


class GWSolver(ContextlessBaseSolver):
    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD


class FGWSolver(ContextlessBaseSolver):
    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD_FUSED

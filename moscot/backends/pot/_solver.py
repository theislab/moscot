from typing import Any, Dict, List, Mapping, Optional
import warnings

from ot import sinkhorn, sinkhorn_unbalanced
from ot.gromov import fused_gromov_wasserstein, entropic_gromov_wasserstein
from ot.backend import get_backend
from typing_extensions import Literal
import ot

import numpy as np
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
        tau_a: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        M = _create_cost_matrix(x, y, **kwargs)
        a = [] if a is None else a
        b = [] if b is None else b
        epsilon = 1e-2 if epsilon is None else epsilon

        res = {"a": a, "b": b, "M": M, "reg": epsilon}
        if tau_a != 1.0:
            res["reg_m"] = -(tau_a * epsilon) / (tau_a - 1)
        return res

    def _solve(self, data: Mapping[str, Any], **kwargs: Any) -> BaseSolverOutput:
        kwargs["log"] = True  # for error
        kwargs["warn"] = True  # for convergence
        solver = sinkhorn_unbalanced if "reg_m" in data else sinkhorn
        with warnings.catch_warnings(record=True) as messages:
            T, log = solver(**data, **kwargs)
        cost = log["err"][-1]
        return POTOutput(T, cost=cost, converged=_get_convergence(messages))

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.LINEAR


class GWSolver(ContextlessBaseSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        epsilon: Optional[float] = None,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        loss_fun: Literal["square_loss", "kl_loss"] = "square_loss",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        nx = get_backend(x.data)

        C1 = _create_cost_matrix(x, y=None, **kwargs)
        C2 = _create_cost_matrix(y, y=None, **kwargs)
        a = nx.ones((C1.shape[0],)) if a is None else a
        b = nx.ones((C2.shape[0],)) if b is None else b
        # must be balanced
        a /= nx.sum(a)
        b /= nx.sum(b)
        epsilon = 1e-2 if epsilon is None else epsilon

        return {"C1": C1, "C2": C2, "p": a, "q": b, "epsilon": epsilon, "loss_fun": loss_fun}

    def _solve(self, data: Mapping[str, Any], **kwargs: Any) -> BaseSolverOutput:
        kwargs["log"] = True
        T, log = entropic_gromov_wasserstein(**data, **kwargs)
        cost = log["gw_dist"]
        # TODO(michalk8): there's no convergence warning/flag in log that sets whether we converged...
        return POTOutput(T, cost=cost, converged=np.isfinite(cost))

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD


class FGWSolver(GWSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        xx: Optional[TaggedArray] = None,
        yy: Optional[TaggedArray] = None,
        epsilon: Optional[float] = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not 0 < alpha < 1:
            raise ValueError(f"TODO: invalid alpha value `{alpha}`")
        data = super()._prepare_input(x=x, y=y, epsilon=epsilon, **kwargs)
        data["M"] = _create_cost_matrix(xx, y=yy, **kwargs)
        data["alpha"] = alpha
        # TODO(michalk8): reimplement with epsilon a la `novosparc`
        _ = data.pop("epsilon")

        return data

    def _solve(self, data: Mapping[str, Any], **kwargs: Any) -> BaseSolverOutput:
        kwargs["log"] = True
        T, log = fused_gromov_wasserstein(**data, **kwargs)
        cost = log["fgw_dist"]
        # TODO(michalk8): there's no convergence warning/flag in log that sets whether we converged...
        return POTOutput(T, cost=cost, converged=np.isfinite(cost))

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD_FUSED


def _create_cost_matrix(
    x: TaggedArray,
    y: Optional[TaggedArray] = None,
    p: float = 2.0,
    w: Optional[npt.ArrayLike] = None,
    **_: Any,
) -> npt.ArrayLike:
    metric = "sqeuclidean" if x.loss is None else x.loss
    if y is not None:
        return ot.utils.dist(x.data, y.data, metric=metric, p=p, w=w)
    if x.is_point_cloud:
        return ot.utils.dist(x.data, metric=metric, p=p, w=w)
    if x.is_cost_matrix:
        return x.data
    if x.is_kernel:
        raise NotImplementedError("TODO: POT kernel not implemented")
    raise NotImplementedError("TODO: invalid tag")


def _get_convergence(messages: List[warnings.WarningMessage]) -> bool:
    for message in messages:
        try:
            if "Sinkhorn did not converge" in message.message.args[0]:
                return False
        except IndexError:
            pass
    return True

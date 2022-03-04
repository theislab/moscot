from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, List, Tuple, Union, Mapping, Optional

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.problems import GeneralProblem

__all__ = ("MultiMarginalProblem",)

from moscot.solvers._output import BaseSolverOutput

Marginals_t = Tuple[Optional[np.ndarray], Optional[np.ndarray]]


class MultiMarginalProblem(GeneralProblem, ABC):
    _a: Optional[List[np.ndarray]]
    _b: Optional[List[np.ndarray]]

    @abstractmethod
    def _estimate_marginals(self, adata: AnnData, *, source: bool, **kwargs: Any) -> Optional[npt.ArrayLike]:
        pass

    def prepare(
        self,
        x: Mapping[str, Any] = MappingProxyType({}),
        y: Optional[Mapping[str, Any]] = None,
        xy: Optional[Mapping[str, Any]] = None,
        a: Optional[Union[bool, str, npt.ArrayLike]] = True,
        b: Optional[Union[bool, str, npt.ArrayLike]] = True,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "MultiMarginalProblem":
        # TODO(michalk8): some sentinel value would be nicer
        if a is True:
            a = self._estimate_marginals(self.adata, source=True, **marginal_kwargs)
        if b is True:
            b = self._estimate_marginals(self._marginal_b_adata, source=False, **marginal_kwargs)

        super().prepare(x, y, xy, a=a, b=b, **kwargs)
        # base problem prepare array-like structure, just wrap it
        # alt. we could just append and not reset
        self._a = [self._a]
        self._b = [self._b]

        return self

    def solve(
        self,
        epsilon: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        n_iters: int = 1,
        reset_marginals: bool = True,
        **kwargs: Any,
    ) -> "MultiMarginalProblem":
        assert n_iters > 0, "TODO: Number of iterations must be > 0."
        if reset_marginals:
            self._reset_marginals()

        # TODO(michalk8): keep?
        # set this after the 1st run so that user can ignore the 1st marginals (for consistency with GeneralProblem)
        a, b = self._get_last_marginals()
        a = a / float(np.sum(a))
        b = b / float(np.sum(b))
        kwargs.setdefault("a", a)
        kwargs.setdefault("b", b)

        for _ in range(n_iters):
            sol = super().solve(epsilon=epsilon, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs).solution
            self._add_marginals(sol)
            kwargs["a"], kwargs["b"] = self._get_last_marginals()

        return self

    def _reset_marginals(self) -> None:
        self._a = [] if self._a is None or not len(self._a) else [self._a[0]]
        self._b = [] if self._b is None or not len(self._b) else [self._b[0]]

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        self._a.append(np.asarray(sol.a))
        self._b.append(np.asarray(sol.b))

    def _get_last_marginals(self) -> Marginals_t:
        # solvers are expected to handle `None` as marginals
        a = self._a[-1] if len(self._a) else None
        b = self._b[-1] if len(self._b) else None
        return a, b

    @property
    def a(self) -> Optional[np.ndarray]:
        if not len(self._a):
            return None
        return np.asarray(self._a)

    @property
    def b(self) -> Optional[np.ndarray]:
        if not len(self._b):
            return None
        return np.asarray(self._b)

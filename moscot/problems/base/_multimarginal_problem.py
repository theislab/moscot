from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, List, Tuple, Union, Mapping, Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot._types import ArrayLike
from moscot._docs import d
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.base._base_problem import OTProblem

__all__ = ["MultiMarginalProblem"]


@d.get_sections(base="MultiMarginalProblem", sections=["Parameters", "Raises"])
@d.dedent
class MultiMarginalProblem(OTProblem, ABC):
    """
    Problem class handling one optimal transport subproblem which allows to iteratively solve the optimal transport map.

    Parameters
    ----------
    %(OTProblem.parameters)s

    Raises
    ------
    %(OTProblem.raises)s
    """

    _a: Optional[List[ArrayLike]]  # type: ignore[assignment]
    _b: Optional[List[ArrayLike]]  # type: ignore[assignment]

    @abstractmethod
    def _estimate_marginals(self, adata: AnnData, *, source: bool, **kwargs: Any) -> Optional[ArrayLike]:
        pass

    def prepare(
        self,
        xy: Optional[Mapping[str, Any]] = None,
        x: Optional[Mapping[str, Any]] = None,
        y: Optional[Mapping[str, Any]] = None,
        a: Optional[Union[bool, str, ArrayLike]] = True,
        b: Optional[Union[bool, str, ArrayLike]] = True,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "MultiMarginalProblem":
        """Prepare MultiMarginalProblem."""
        # TODO(michalk8): some sentinel value would be nicer
        if a is True:
            a = self._estimate_marginals(self.adata, source=True, **marginal_kwargs)
        elif a is False:
            a = None
        if b is True:
            b = self._estimate_marginals(self._adata_y, source=False, **marginal_kwargs)
        elif b is False:
            b = None
        _ = super().prepare(xy=xy, x=x, y=y, a=a, b=b, **kwargs)
        # base problem prepare array-like structure, just wrap it
        # alt. we could just append and not reset
        if TYPE_CHECKING:
            assert isinstance(self._a, np.ndarray)
            assert isinstance(self._b, np.ndarray)
        self._a = [self._a]
        self._b = [self._b]

        return self

    def solve(
        self,
        *args: Any,
        n_iters: int = 1,
        reset_marginals: bool = True,
        **kwargs: Any,
    ) -> "MultiMarginalProblem":
        """Solve MultiMarginalProblem."""
        assert n_iters > 0, "TODO: Number of iterations must be > 0."
        if reset_marginals:
            self._reset_marginals()

        # TODO(michalk8): keep?
        # set this after the 1st run so that user can ignore the 1st marginals (for consistency with OTProblem)
        a, b = self._get_last_marginals()
        kwargs.setdefault("a", a)
        kwargs.setdefault("b", b)

        for _ in range(n_iters):
            sol = super().solve(*args, **kwargs).solution
            if TYPE_CHECKING:
                assert isinstance(sol, BaseSolverOutput)
            self._add_marginals(sol)
            kwargs["a"], kwargs["b"] = self._get_last_marginals()

        return self

    def _reset_marginals(self) -> None:
        self._a = [] if self._a is None or not len(self._a) else [self._a[0]]
        self._b = [] if self._b is None or not len(self._b) else [self._b[0]]

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        if TYPE_CHECKING:
            assert isinstance(self._a, list)
            assert isinstance(self._b, list)
        self._a.append(np.asarray(sol.a))
        self._b.append(np.asarray(sol.b))

    def _get_last_marginals(self) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
        # solvers are expected to handle `None` as marginals
        if TYPE_CHECKING:
            assert isinstance(self._a, list)
            assert isinstance(self._b, list)
        a = self._a[-1] if len(self._a) else None
        b = self._b[-1] if len(self._b) else None
        return a, b

    @property
    def a(self) -> Optional[ArrayLike]:
        """Array of all left marginals."""
        if TYPE_CHECKING:
            assert isinstance(self._a, list)
        return np.asarray(self._a).T if len(self._a) else None

    @property
    def b(self) -> Optional[ArrayLike]:
        """Array of all right marginals."""
        if TYPE_CHECKING:
            assert isinstance(self._b, list)
        return np.asarray(self._b).T if len(self._b) else None

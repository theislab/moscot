from enum import Enum
from typing import Any, Dict, Type, Tuple, Union, Literal, Iterator, Optional, Sequence
from itertools import product

from pandas.api.types import is_categorical_dtype
import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot._base import BaseSolver
from moscot.tmp.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.tmp.solvers._output import BaseSolverOutput
from moscot.tmp.problems._base_problem import BaseProblem, GeneralProblem
from moscot.tmp.problems._subset_policy import SubsetPolicy, ExplicitPolicy


# TODO(michalk8): should be a base class + subclasses + classmethod create
class Policy(str, Enum):
    PAIRWISE = "pairwise"
    SUBSEQUENT = "subsequent"
    UPPER_DIAG = "upper_diag"

    def create(
        self, data: Union[pd.Series, pd.Categorical], subset: Optional[Sequence[Any]] = None
    ) -> Dict[Tuple[Any, Any], Tuple[pd.Series, pd.Series]]:
        # TODO(michalk8): handle explicitly passed policy: must be a sequence of 2-tuples
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        # TODO(michalk8): allow explicit conversion from bools/numeric/strings as long as number of generated
        # categories is sane (e.g. <= 64)?
        if not is_categorical_dtype(data):
            raise TypeError("TODO - expected categorical")
        categories = [c for c in data.cat.categories if subset is None or c in subset]
        if not categories:
            raise ValueError("TODO - no valid subset has been selected.")

        if self.value == "pairwise":
            return {(x, y): (data == x, data == y) for x, y in product(categories, categories)}

        if not data.cat.ordered:
            # TODO(michalk8): use? https://github.com/theislab/cellrank/blob/dev/cellrank/tl/kernels/_utils.py#L255
            raise ValueError("TODO - expected ordered categorical")

        if self.value == "upper_diag":
            return {(x, y): (data == x, data == y) for x, y in product(categories, categories) if x <= y}

        if len(categories) < 2:
            raise ValueError("TODO - subsequent too few categories, point to GeneralProblem")
        return {(x, y): (data == x, data == y) for x, y in zip(categories[:-1], categories[1:])}


# TODO(michalk8): make abstract?
class CompoundProblem(BaseProblem):
    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None):
        super().__init__(adata, solver)

        self._problems: Optional[Dict[Tuple[Any, Any], GeneralProblem]] = None
        self._solutions: Optional[Dict[Tuple[Any, Any], BaseSolverOutput]] = None
        self._subset: Optional[SubsetPolicy] = None

    def prepare(
        self,
        key: str,
        subset: Optional[Sequence[Any]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "BaseProblem":
        if isinstance(policy, str):
            self._subset = SubsetPolicy.create(policy, self.adata, key=key).subset(subset)
        else:  # policy contains the actual pairs
            self._subset = ExplicitPolicy(self.adata, key=key).subset(subset, policy)

        self._problems = {
            subset: GeneralProblem(self.adata[x_mask, :], self.adata[y_mask, :], solver=self._solver).prepare(**kwargs)
            for subset, (x_mask, y_mask) in self._subset.mask(discard_empty=True).items()
        }

        return self

    def solve(
        self,
        eps: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        **kwargs: Any,
    ) -> "BaseProblem":
        self._solutions = {}
        for subset, problem in self._problems.items():
            self._solutions[subset] = problem.solve(eps=eps, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs)

        return self

    def _apply(self, x: npt.ArrayLike, *, forward: bool) -> npt.ArrayLike:
        pass

    def push(
        self,
        key: Union[str, npt.ArrayLike],
        subset: Optional[Sequence[Any]] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        normalize: bool = True,
    ) -> npt.ArrayLike:
        if isinstance(key, str):
            key = ...
        key = np.asarray(key)

    @property
    def _valid_solver_types(self) -> Tuple[Type[BaseSolver], ...]:
        return SinkhornSolver, GWSolver, FGWSolver

    @property
    def solution(self) -> Optional[Dict[Tuple[Any, Any], BaseSolverOutput]]:
        return self._solutions

    def __getitem__(self, item: Tuple[Any, Any]) -> BaseSolverOutput:
        return self.solution[item]

    def __len__(self) -> int:
        return 0 if self.solution is None else len(self.solution)

    def __iter__(self) -> Iterator:
        if self.solution is None:
            raise StopIteration
        return iter(self.solution.items())

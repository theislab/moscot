from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Tuple, Union, Literal, Optional, Sequence

import pandas as pd

from anndata import AnnData

from moscot._base import BaseSolver
from moscot.tmp.solvers._output import BaseSolverOutput
from moscot.tmp.problems._base_problem import BaseProblem


class Policy(str, Enum):
    PAIRWISE = "pairwise"
    SUBSEQUENT = "subsequent"
    UPPER_DIAG = "upper_diag"

    def create(
        self, data: Union[pd.Series, pd.Categorical], subset: Optional[Sequence[Any]] = None
    ) -> List[Tuple[Any, Any]]:
        pass


class CompoundProblem(BaseProblem, ABC):
    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None):
        super().__init__(adata, solver)

        self._data: Optional[Dict[Tuple[Any, Any]], Any] = None
        self._solution: Optional[Dict[Tuple[Any, Any]], BaseSolverOutput] = None

    def prepare(
        self,
        key: str,
        subset: Optional[Sequence[Any]] = None,
        policy: Literal["pairwise", "subsequent", "upper_diag"] = "pairwise",
        **kwargs: Any,
    ) -> "BaseProblem":
        policy = Policy(policy)
        subsets = policy.create(self.adata.obs[key], subset=subset)
        self._data = {subset: "TODO" for subset in subsets}

        return self

    def solve(self, eps: Optional[float] = None, alpha: Optional[float] = None) -> "BaseProblem":
        self._solution = {}
        for subset, data in self._data.items():
            self._solution[subset] = self._solver(**data, eps=eps)

        return self

    @property
    def solution(self) -> Any:
        return self._solution

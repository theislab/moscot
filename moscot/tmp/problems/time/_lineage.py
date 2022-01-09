from typing import Any, Type, Tuple

from moscot.tmp.mixins import TemporalAnalysisMixin
from moscot.tmp.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.tmp.solvers._base_solver import BaseSolver
from moscot.tmp.problems._base_problem import BaseProblem


class TemporalProblem(TemporalAnalysisMixin, BaseProblem):
    @property
    def _valid_solver_types(self) -> Tuple[Type[BaseSolver], ...]:
        return SinkhornSolver, GWSolver, FGWSolver

    def prepare(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        pass

    def solve(self) -> "BaseProblem":
        pass

    @property
    def solution(self) -> Any:
        pass

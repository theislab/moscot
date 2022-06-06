from types import MappingProxyType
from typing import Dict, Tuple, Union, Generic, Mapping, TypeVar, Hashable, Optional
from collections import defaultdict

from moscot.solvers._output import BaseSolverOutput
from moscot.problems._subset_policy import SubsetPolicy
from moscot.problems.base._base_problem import OTProblem, ProblemStage

K = TypeVar("K", bound=Hashable)
B = TypeVar("B", bound=OTProblem)


class ProblemManager(Generic[K, B]):
    def __init__(self, policy: SubsetPolicy[K], problems: Mapping[Tuple[K, K], B] = MappingProxyType({})):
        self._policy = policy
        self._problems: Dict[Tuple[K, K], B] = dict(problems)
        self._verify_shape_integrity()

    def add_problem(self, key: Tuple[K, K], problem: B, *, overwrite: bool = False) -> None:
        self._add_problem(key, problem, overwrite=overwrite, verify_integrity=True)

    def _add_problem(
        self, key: Tuple[K, K], problem: B, *, overwrite: bool = False, verify_integrity: bool = True
    ) -> None:
        if not overwrite and key in self.problems:
            raise KeyError(f"TODO: `{key}` already present, use `overwrite=True`")
        self.problems[key] = problem
        self._policy.add_node(key)
        if verify_integrity:
            self._verify_shape_integrity()

    def add_problems(self, problems: Dict[Tuple[K, K], B], overwrite: bool = True) -> None:
        for key, prob in problems.items():
            self._add_problem(key, prob, overwrite=overwrite, verify_integrity=False)
        self._verify_shape_integrity()

    def remove_problem(self, key: Tuple[K, K]) -> None:
        del self.problems[key]
        self._policy.remove_node(key)

    def get_problems(
        self, stage: Optional[Union[ProblemStage, Tuple[ProblemStage, ...]]] = None
    ) -> Dict[Tuple[K, K], B]:
        if stage is None:
            return self._problems
        if isinstance(stage, ProblemStage):
            stage = (stage,)
        return {k: v for k, v in self.problems.items() if v.stage in stage}

    def get_solutions(self, only_converged: bool = False) -> Dict[Tuple[K, K], BaseSolverOutput]:
        return {
            k: v.solution
            for k, v in self.problems.items()
            if v.solution is not None and (not only_converged or v.solution.converged)
        }

    def _verify_shape_integrity(self) -> None:
        dims = defaultdict(set)
        for (src, tgt), prob in self.problems.items():
            dims[src].add(prob.shape[0])
            dims[tgt].add(prob.shape[1])

        for key, dim in dims.items():
            if len(dim) > 1:
                raise ValueError(f"TODO: key `{key}` is associated with more than 1 dimensnions `{dim}`")

    @property
    def solutions(self) -> Dict[Tuple[K, K], BaseSolverOutput]:
        return self.get_solutions(only_converged=False)

    @property
    def problems(self) -> Dict[Tuple[K, K], B]:
        return self._problems

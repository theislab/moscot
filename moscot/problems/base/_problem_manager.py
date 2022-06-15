from types import MappingProxyType
from typing import Any, Dict, Tuple, Union, Generic, Mapping, TypeVar, Hashable, Optional, TYPE_CHECKING
from collections import defaultdict

from moscot.solvers._output import BaseSolverOutput
from moscot.problems._subset_policy import SubsetPolicy
from moscot.problems.base._base_problem import OTProblem, ProblemStage

if TYPE_CHECKING:
    from moscot.problems.base._compound_problem import BaseCompoundProblem

K = TypeVar("K", bound=Hashable)
B = TypeVar("B", bound=OTProblem)


class ProblemManager(Generic[K, B]):
    """Class handling problem policies."""

    def __init__(self, compound_problem: "BaseCompoundProblem[K, B]", policy: SubsetPolicy[K]):
        self._compound_problem = compound_problem
        self._policy = policy
        self._problems: Dict[Tuple[K, K], B] = {}

    def add_problem(self, key: Tuple[K, K], problem: Optional[B] = None, *, overwrite: bool = False) -> None:
        """Add problem."""
        self._add_problem(key, problem, overwrite=overwrite, verify_integrity=True)

    def _add_problem(
        self,
        key: Tuple[K, K],
        problem: Optional[B] = None,
        *,
        overwrite: bool = False,
        verify_integrity: bool = True,
        **kwargs: Any,
    ) -> None:
        from moscot.problems.base._compound_problem import CompoundProblem

        if not overwrite and key in self.problems:
            raise KeyError(f"TODO: `{key}` already present, use `overwrite=True`")

        if problem is None:
            problem = self._create_problem(key, **kwargs)

        if isinstance(self._compound_problem, CompoundProblem):
            if not isinstance(problem, self._compound_problem._base_problem_type):
                raise TypeError(f"TODO: expected `{self._compound_problem._base_problem_type}`, got `{type(problem)}`")
        elif not isinstance(problem, OTProblem):
            raise TypeError(f"TODO: expected `{OTProblem}`, got `{type(problem)}`")

        self.problems[key] = problem
        self._policy.add_node(key)
        if verify_integrity:
            self._verify_shape_integrity()

    def add_problems(self, problems: Dict[Tuple[K, K], Optional[B]], overwrite: bool = True) -> None:
        """Add problems."""
        for key, prob in problems.items():
            self._add_problem(key, prob, overwrite=overwrite, verify_integrity=False)
        self._verify_shape_integrity()

    def remove_problem(self, key: Tuple[K, K]) -> None:
        """Remove a problem."""
        del self.problems[key]
        self._policy.remove_node(key)

    def get_problems(
        self, stage: Union[ProblemStage, Tuple[ProblemStage, ...]] = (ProblemStage.PREPARED, ProblemStage.SOLVED)
    ) -> Dict[Tuple[K, K], B]:
        """Get problems."""
        if stage is None:
            return self._problems
        if isinstance(stage, ProblemStage):
            stage = (stage,)
        return {k: v for k, v in self.problems.items() if v.stage in stage}

    def get_solutions(self, only_converged: bool = False) -> Dict[Tuple[K, K], BaseSolverOutput]:
        """Return solutions."""
        return {
            k: v.solution
            for k, v in self.problems.items()
            if v.solution is not None and (not only_converged or v.solution.converged)
        }

    def _create_problem(
        self, key: Tuple[K, K], init_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any
    ) -> B:
        src_mask = self._policy.create_mask(key[0], allow_empty=False)
        tgt_mask = self._policy.create_mask(key[1], allow_empty=False)
        return self._compound_problem._create_problem(src_mask, tgt_mask, **init_kwargs).prepare(**kwargs)

    def _verify_shape_integrity(self) -> None:
        dims = defaultdict(set)
        for (src, tgt), prob in self.problems.items():
            dims[src].add(prob.shape[0])
            dims[tgt].add(prob.shape[1])

        for key, dim in dims.items():
            if len(dim) > 1:
                raise ValueError(f"TODO: key `{key}` is associated with more than 1 dimensnions `{dim}`")

    @property
    def solutions(self) -> Dict[Tuple[K, K], BaseSolverOutput]:  # noqa: D102
        return self.get_solutions(only_converged=False)

    @property
    def problems(self) -> Dict[Tuple[K, K], B]:  # noqa: D102
        return self._problems

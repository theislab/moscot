import collections
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    Hashable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from moscot._types import ProblemStage_t
from moscot.base.output import BaseSolverOutput
from moscot.base.problems.problem import OTProblem
from moscot.utils.subset_policy import SubsetPolicy

if TYPE_CHECKING:
    from moscot.base.problems.compound_problem import BaseCompoundProblem

K = TypeVar("K", bound=Hashable)
B = TypeVar("B", bound=OTProblem)

__all__ = ["ProblemManager"]


class ProblemManager(Generic[K, B]):
    """Manager which helps to add and remove problems based on the :attr:`policy`.

    Parameters
    ----------
    compound_problem
        Problem containing multiple subproblems.
    policy
        Subset policy guiding this manager.
    """

    def __init__(self, compound_problem: "BaseCompoundProblem[K, B]", policy: SubsetPolicy[K]):
        self._compound_problem = compound_problem
        self._policy = policy
        self._problems: Dict[Tuple[K, K], B] = {}

    def add_problem(
        self, key: Tuple[K, K], problem: B, *, overwrite: bool = False, verify_integrity: bool = True
    ) -> None:
        """Add a subproblem.

        Parameters
        ----------
        key
            Key in :attr:`problems` where to add the subproblem.
        problem
            Subproblem to add.
        overwrite
            Whether to overwrite an existing problem.
        verify_integrity
            Whether to check if the ``problem`` has the correct shape.

        Returns
        -------
        Nothing, just updates the following fields:

        - :attr:`problems['{key}']` - the added subproblem.
        """
        from moscot.base.problems.compound_problem import CompoundProblem

        if not overwrite and key in self.problems:
            raise KeyError(f"Problem `{key}` is already present, use `overwrite=True` to add it.")

        clazz = (
            self._compound_problem._base_problem_type
            if isinstance(self._compound_problem, CompoundProblem)
            else OTProblem
        )
        if not isinstance(problem, clazz):  # type:ignore[arg-type]
            raise TypeError(f"Expected problem of type `{OTProblem}`, found `{type(problem)}`.")

        self.problems[key] = problem
        self.policy.add_node(key)
        if verify_integrity:
            self._verify_shape_integrity()
            # TODO(michalk8): add check for obs/var names

    def add_problems(
        self, problems: Dict[Tuple[K, K], B], overwrite: bool = False, verify_integrity: bool = True
    ) -> None:
        """Add multiple subproblems in bulk.

        Parameters
        ----------
        problems
            Subproblems to add.
        overwrite
            Whether to overwrite existing keys in :attr:`problems`.
        verify_integrity
            Whether to check the ``problems`` have the correct shape.

        Returns
        -------
        Nothing, just adds the subproblems to :attr:`problems`.
        """
        for key, prob in problems.items():
            self.add_problem(key, prob, overwrite=overwrite, verify_integrity=False)
        if verify_integrity:
            self._verify_shape_integrity()

    def remove_problem(self, key: Tuple[K, K]) -> None:
        """Remove a subproblem.

        Parameters
        ----------
        key
            Key of the subproblem to remove.

        Returns
        -------
        Nothing, just removes the subproblem from :attr:`problem`.

        Raises
        ------
        KeyError
            If the ``key`` is not in :attr:`problems`.
        """
        del self.problems[key]
        self.policy.remove_node(key)

    def get_problems(
        self,
        stage: Optional[Union[ProblemStage_t, Tuple[ProblemStage_t, ...]]] = None,
    ) -> Dict[Tuple[K, K], B]:
        """Get the :term:`OT` subproblems.

        Parameters
        ----------
        stage
            Problem stage used for filtering. If :obj:`None`, return all :attr:`problems`.

        Returns
        -------
        :term:`OT` problems filtered by their :attr:`~moscot.base.problems.BaseProblem.stage`.
        """
        if stage is None:
            return self.problems
        stage = (stage,) if isinstance(stage, str) else stage
        return {k: v for k, v in self.problems.items() if v.stage in stage}

    def get_solutions(self, only_converged: bool = False) -> Dict[Tuple[K, K], BaseSolverOutput]:
        """Get solutions to the :term:`OT` subproblems.

        Parameters
        ----------
        only_converged
            Whether to return only converged solutions.

        Returns
        -------
        The :term:`OT` solutions for :attr:`problems`.
        """
        return {
            k: v.solution
            for k, v in self.problems.items()
            if v.solution is not None and (not only_converged or v.solution.converged)
        }

    def _verify_shape_integrity(self) -> None:
        dims = collections.defaultdict(set)
        for (src, tgt), prob in self.problems.items():
            n, m = prob.shape
            dims[src].add(n)
            dims[tgt].add(m)

        for key, dim in dims.items():
            if len(dim) > 1:
                raise ValueError(f"Problem `{key}` is associated with different dimensions: `{dim}`.")

    @property
    def solutions(self) -> Dict[Tuple[K, K], BaseSolverOutput]:
        """Solutions for the :term:`OT` :attr:`problems`."""
        return self.get_solutions(only_converged=False)

    @property
    def problems(self) -> Dict[Tuple[K, K], B]:
        """:term:`OT` problems."""
        return self._problems

    @property
    def policy(self) -> SubsetPolicy[K]:
        """Policy guiding this manager."""
        return self._policy

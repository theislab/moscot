import collections
from typing import TYPE_CHECKING, Dict, Generic, Hashable, Tuple, TypeVar, Union

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
        Policy that defined how individual problems in ``compound_problem`` are constructed.
    """

    def __init__(self, compound_problem: "BaseCompoundProblem[K, B]", policy: SubsetPolicy[K]):
        self._compound_problem = compound_problem
        self._policy = policy
        self._problems: Dict[Tuple[K, K], B] = {}

    def add_problem(
        self, key: Tuple[K, K], problem: B, *, overwrite: bool = False, verify_integrity: bool = True
    ) -> None:
        """Add a problem to :attr:`problems`.

        Parameters
        ----------
        key
            Key of the problem.
        problem
            Problem to add.
        overwrite
            Whether to overwrite an existing problem.
        verify_integrity
            Whether to check the ``problem`` is compatible with the :attr:`policy`.

        Returns
        -------
        Nothing, just adds the ``problem`` to :attr:`problems`.
        """
        from moscot.base.problems.compound_problem import CompoundProblem

        if not overwrite and key in self.problems:
            raise KeyError(f"Problem `{key}` is already present, use `overwrite=True` to add it.")

        clazz = (
            self._compound_problem._base_problem_type
            if isinstance(self._compound_problem, CompoundProblem)
            else OTProblem
        )
        if not isinstance(problem, clazz):
            raise TypeError(f"Expected problem of type `{OTProblem}`, found `{type(problem)}`.")

        self.problems[key] = problem
        self.policy.add_node(key)
        if verify_integrity:
            self._verify_shape_integrity()

    def add_problems(self, problems: Dict[Tuple[K, K], B], overwrite: bool = False) -> None:
        """Add multiple problems to :attr:`problems`.

        Parameters
        ----------
        problems
            Problems to add.
        overwrite
            Whether to overwrite existing problems.

        Returns
        -------
        Nothing, just adds the ``problems`` to :attr:`problems`.
        """
        for key, prob in problems.items():
            self.add_problem(key, prob, overwrite=overwrite, verify_integrity=False)
        self._verify_shape_integrity()

    def remove_problem(self, key: Tuple[K, K]) -> None:
        """Remove a problem from :attr:`problems`.

        Parameters
        ----------
        key
            Key of the problem to remove.

        Returns
        -------
        Nothing, just removes the problem from :attr:`problem`.

        Raises
        ------
        KeyError
            If the problem is not in :attr:`problems`.
        """
        del self.problems[key]
        self.policy.remove_node(key)

    def get_problems(
        self,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
    ) -> Dict[Tuple[K, K], B]:
        """Get :attr:`problems` filtered by their stage.

        Parameters
        ----------
        stage
            Problem staged used for filtering.

        Returns
        -------
        :term:`OT` problems with the above-mentioned ``stage``.
        """
        if stage is None:
            return self.problems
        stage = (stage,) if isinstance(stage, str) else stage
        return {k: v for k, v in self.problems.items() if v.stage in stage}

    def get_solutions(self, only_converged: bool = False) -> Dict[Tuple[K, K], BaseSolverOutput]:
        """Get solutions to :attr:`problems`, if present.

        Parameters
        ----------
        only_converged
            Whether to return only converged solutions.

        Returns
        -------
        Solution for each problem
        """
        return {
            k: v.solution
            for k, v in self.problems.items()
            if v.solution is not None and (not only_converged or v.solution.converged)
        }

    def _verify_shape_integrity(self) -> None:
        # TODO(michalk8): check whether the `AnnData`'s indices are aligned
        dims = collections.defaultdict(set)
        for (src, tgt), prob in self.problems.items():
            dims[src].add(prob.shape[0])
            dims[tgt].add(prob.shape[1])

        for key, dim in dims.items():
            if len(dim) > 1:
                raise ValueError(f"Problem `{key}` is associated with different dimensions: `{dim}`.")

    @property
    def solutions(self) -> Dict[Tuple[K, K], BaseSolverOutput]:
        """All :term:`OT` solutions for the given :attr:`problems`."""
        return self.get_solutions(only_converged=False)

    @property
    def problems(self) -> Dict[Tuple[K, K], B]:
        """All :term:`OT` problems."""
        return self._problems

    @property
    def policy(self) -> SubsetPolicy[K]:
        """Policy guiding this manager."""
        return self._policy

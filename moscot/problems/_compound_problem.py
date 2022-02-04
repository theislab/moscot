from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Tuple, Union, Literal, Mapping, Iterator, Optional, Sequence

import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import BaseSolver
from moscot.problems._base_problem import BaseProblem, GeneralProblem
from moscot.problems._subset_policy import StarPolicy, SubsetPolicy, ExplicitPolicy

__all__ = ("SingleCompoundProblem", "MultiCompoundProblem", "CompoundProblem")


class CompoundBaseProblem(BaseProblem, ABC):
    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None):
        super().__init__(adata, solver)

        self._problems: Optional[Dict[Tuple[Any, Any], GeneralProblem]] = None
        self._solutions: Optional[Dict[Tuple[Any, Any], BaseSolverOutput]] = None
        self._policy: Optional[SubsetPolicy] = None

    @abstractmethod
    def _create_problems(self, **kwargs: Any) -> Dict[Tuple[Any, Any], GeneralProblem]:
        pass

    @abstractmethod
    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> SubsetPolicy:
        pass

    def prepare(
        self,
        key: str,
        subset: Optional[Sequence[Any]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        reference: Optional[Any] = None,
        **kwargs: Any,
    ) -> "CompoundProblem":
        policy = self._create_policy(policy=policy, key=key)

        if isinstance(policy, ExplicitPolicy):
            policy = policy.subset(subset, policy)
        elif isinstance(policy, StarPolicy):
            if reference is None:
                raise ValueError("TODO: specify star reference")
            policy = policy.subset(subset, reference=reference)
        else:
            policy = policy.subset(subset)

        self._policy = policy
        self._problems = self._create_problems(**kwargs)

        return self

    def solve(
        self,
        eps: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        **kwargs: Any,
    ) -> "CompoundProblem":
        self._solutions = {}
        for subset, problem in self._problems.items():
            self._solutions[subset] = problem.solve(eps=eps, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs)

        return self

    def _apply(
        self,
        data: Optional[Union[str, npt.ArrayLike]] = None,
        subset: Optional[Sequence[Any]] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        normalize: bool = True,
        return_all: bool = False,
        forward: bool = True,
    ) -> Union[npt.ArrayLike, npt.ArrayLike]:
        # TODO: check if solved - decorator?
        # TODO(michalk8): default should be policy dependent (star/seq esp.)
        if forward:
            pairs = self._policy.chain(start, end)
        else:
            # TODO: mb. don't swap start/end
            # start, end = end, start
            pairs = self._policy.chain(start, end)[::-1]

        if return_all:
            problem = self._problems[pairs[0]]
            adata = problem.adata if forward or problem._adata_y is None else problem._adata_y
            data = [problem._get_mass(adata, data, subset=subset, normalize=True)]
        else:
            data = [data]

        for pair in pairs:
            problem = self._problems[pair]
            data.append((problem.push if forward else problem.pull)(data[-1], subset=subset, normalize=normalize))
            if not return_all:
                data = [data[-1]]

        return data if return_all else data[-1]

    def push(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        return self._apply(*args, forward=True, **kwargs)

    def pull(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        return self._apply(*args, forward=False, **kwargs)

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


class SingleCompoundProblem(CompoundBaseProblem):
    def _create_problems(self, **kwargs: Any) -> Dict[Tuple[Any, Any], GeneralProblem]:
        return {
            subset: GeneralProblem(self.adata[x_mask, :], self.adata[y_mask, :], solver=self._solver).prepare(**kwargs)
            for subset, (x_mask, y_mask) in self._policy.mask(discard_empty=True).items()
        }

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        key: Optional[str] = None,
        **_: Any,
    ) -> SubsetPolicy:
        return (
            SubsetPolicy.create(policy, self.adata, key=key)
            if isinstance(policy, str)
            else ExplicitPolicy(self.adata, key=key)
        )


class MultiCompoundProblem(CompoundBaseProblem):
    _KEY = "subset"

    def __init__(
        self,
        *adatas: Union[AnnData, Mapping[Any, AnnData], Tuple[AnnData], List[AnnData]],
        solver: Optional[BaseSolver] = None,
    ):
        if not len(adatas):
            raise ValueError("TODO: no adatas passed")

        if len(adatas) == 1:
            if isinstance(adatas[0], Mapping):
                adata = next(iter(adatas[0].values()))
                adatas = adatas[0]
            elif isinstance(adatas[0], AnnData):
                adata = adatas[0]
            elif isinstance(adatas[0], (tuple, list)):
                adatas = adatas[0]
                adata = adatas[0]
            else:
                raise TypeError("TODO: no adatas passed")
        else:
            adata = adatas[0]

        # TODO(michalk8): can this have unintended consequences in push/pull?
        super().__init__(adata, solver)

        if not isinstance(adatas, Mapping):
            adatas = {i: adata for i, adata in enumerate(adatas)}

        self._adatas: Mapping[Any, AnnData] = adatas
        self._policy_adata = AnnData(
            np.empty((len(self._adatas), 1)),
            obs=pd.DataFrame({self._KEY: pd.Series(list(self._adatas.keys())).astype("category")}),
        )

    def prepare(
        self,
        subset: Optional[Sequence[Any]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        reference: Optional[Any] = None,
        **kwargs: Any,
    ) -> "MultiCompoundProblem":
        return super().prepare(None, subset=subset, policy=policy, reference=reference, **kwargs)

    def _create_problems(self, **kwargs: Any) -> Dict[Tuple[Any, Any], GeneralProblem]:
        return {
            (x, y): GeneralProblem(self._adatas[x], self._adatas[y], solver=self._solver).prepare(**kwargs)
            for x, y in self._policy.mask(discard_empty=True).keys()
        }

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **_: Any,
    ) -> SubsetPolicy:
        return (
            SubsetPolicy.create(policy, self._policy_adata, key=self._KEY)
            if isinstance(policy, str)
            else ExplicitPolicy(self._policy_adata, key=self._KEY)
        )


class CompoundProblem(CompoundBaseProblem):
    def __init__(
        self,
        *adatas: Union[AnnData, Mapping[Any, AnnData], Tuple[AnnData], List[AnnData]],
        solver: Optional[BaseSolver] = None,
    ):
        if len(adatas) == 1 and isinstance(adatas[0], AnnData):
            self._prob = SingleCompoundProblem(adatas[0], solver=solver)
        else:
            self._prob = MultiCompoundProblem(*adatas, solver=solver)

        super().__init__(self._prob.adata, self._prob._solver)

    def _create_problems(self, **kwargs: Any) -> Dict[Tuple[Any, Any], GeneralProblem]:
        return self._prob._create_problems(**kwargs)

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        key: Optional[str] = None,
        **_: Any,
    ) -> SubsetPolicy:
        self._prob._policy = self._prob._create_policy(policy=policy, key=key)
        return self._prob._policy

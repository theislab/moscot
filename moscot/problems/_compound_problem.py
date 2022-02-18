from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Tuple, Union, Mapping, Iterator, Optional, Sequence

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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
    def __init__(
        self,
        adata: AnnData,
        solver: Optional[BaseSolver] = None,
        *,
        base_problem_type: Type[BaseProblem] = GeneralProblem,
    ):
        super().__init__(adata, solver)

        self._problems: Optional[Dict[Tuple[Any, Any], BaseProblem]] = None
        self._solutions: Optional[Dict[Tuple[Any, Any], BaseSolverOutput]] = None
        self._policy: Optional[SubsetPolicy] = None
        if not issubclass(base_problem_type, BaseProblem):
            raise TypeError("TODO: `base_problem_type` must be a subtype of `BaseProblem`.")
        self._base_problem_type = base_problem_type

    @abstractmethod
    def _create_problems(self, **kwargs: Any) -> Dict[Tuple[Any, Any], BaseProblem]:
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
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        subset: Optional[Sequence[Tuple[Any, Any]]] = None,
        reference: Optional[Any] = None,
        **kwargs: Any,
    ) -> "CompoundProblem":
        self._policy = self._create_policy(policy=policy, key=key)

        if isinstance(self._policy, ExplicitPolicy):
            self._policy = self._policy(policy)
        elif isinstance(self._policy, StarPolicy):
            self._policy = self._policy(filter=subset, reference=reference)
        else:
            self._policy = self._policy(filter=subset)

        self._problems = self._create_problems(**kwargs)
        self._solutions = None

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
        data: Optional[Union[str, npt.ArrayLike, Mapping[Tuple[Any, Any], Union[str, npt.ArrayLike]]]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
        forward: bool = True,
        return_all: bool = False,
        scale_by_marginals: bool = False,
        **kwargs: Any,
    ) -> Dict[Tuple[Any, Any], npt.ArrayLike]:
        def get_data(plan: Tuple[Any, Any]) -> Optional[npt.ArrayLike]:
            if data is None or isinstance(data, (str, tuple, list)):
                # always valid shapes, since accessing AnnData
                return data
            if isinstance(data, Mapping):
                return data.get(plan[0], None) if isinstance(self._policy, StarPolicy) else data.get(plan, None)
            if len(plans) == 1:
                return data
            # TODO(michalk8): warn
            # would pass an array that will most likely have invalid shapes
            print("HINT: use `data={<pair>: array}`")
            return None

        # TODO: check if solved - decorator?

        plans = self._policy.plan(**kwargs)
        res: Dict[Tuple[Any, Any], npt.ArrayLike] = {}

        for plan, steps in plans.items():
            if not forward:
                steps = steps[::-1]

            ds = [get_data(plan)]
            for step in steps:
                problem = self._problems[step]
                fun = problem.push if forward else problem.pull
                ds.append(fun(ds[-1], subset=subset, normalize=normalize, scale_by_marginals=scale_by_marginals))

            # TODO(michalk8): shall we include initial input? or add as option?
            res[plan] = ds[1:] if return_all else ds[-1]

        # TODO(michalk8): return the values iff only 1 plan?
        return res

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
    def _create_problems(self, **kwargs: Any) -> Dict[Tuple[Any, Any], BaseProblem]:
        return {
            (x, y): self._base_problem_type(
                self._mask(x, x_mask, self._adata_src), self._mask(y, y_mask, self._adata_tgt), solver=self._solver
            ).prepare(**kwargs)
            for (x, y), (x_mask, y_mask) in self._policy.mask().items()
        }

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit", "external_star"] = "sequential",
        key: Optional[str] = None,
        **_: Any,
    ) -> SubsetPolicy:
        return (
            SubsetPolicy.create(policy, self.adata, key=key)
            if isinstance(policy, str)
            else ExplicitPolicy(self.adata, key=key)
        )

    def _mask(self, key: Any, mask: npt.ArrayLike, adata: AnnData) -> AnnData:
        # TODO(michalk8): can include logging/extra sanity that mask is not empty
        return adata[mask]

    @property
    def _adata_src(self) -> AnnData:
        return self.adata

    @property
    def _adata_tgt(self) -> AnnData:
        return self.adata


class MultiCompoundProblem(CompoundBaseProblem):
    _KEY = "subset"

    def __init__(
        self,
        *adatas: Union[AnnData, Mapping[Any, AnnData], Tuple[AnnData], List[AnnData]],
        solver: Optional[BaseSolver] = None,
        **kwargs: Any,
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
        super().__init__(adata, solver, **kwargs)

        if not isinstance(adatas, Mapping):
            adatas = {i: adata for i, adata in enumerate(adatas)}

        self._adatas: Mapping[Any, AnnData] = adatas
        # TODO (ZP): raises a warning
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

    def _create_problems(self, **kwargs: Any) -> Dict[Tuple[Any, Any], BaseProblem]:
        return {
            (x, y): self._base_problem_type(self._adatas[x], self._adatas[y], solver=self._solver).prepare(**kwargs)
            for x, y in self._policy.mask().keys()
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
        **kwargs: Any,
    ):
        if len(adatas) == 1 and isinstance(adatas[0], AnnData):
            self._prob = SingleCompoundProblem(adatas[0], solver=solver, **kwargs)
        else:
            self._prob = MultiCompoundProblem(*adatas, solver=solver, **kwargs)

        super().__init__(self._prob.adata, self._prob._solver)

    def _create_problems(self, **kwargs: Any) -> Dict[Tuple[Any, Any], BaseProblem]:
        return self._prob._create_problems(**kwargs)

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        key: Optional[str] = None,
        **_: Any,
    ) -> SubsetPolicy:
        self._prob._policy = self._prob._create_policy(policy=policy, key=key)
        return self._prob._policy

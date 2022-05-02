from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    List,
    Type,
    Tuple,
    Union,
    Generic,
    Literal,
    Mapping,
    TypeVar,
    Callable,
    Iterator,
    Optional,
    Sequence,
)

import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.backends.ott import SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import OTSolver, ProblemKind
from moscot.problems._base_problem import OTProblem, BaseProblem
from moscot.problems._subset_policy import Axis_t, StarPolicy, SubsetPolicy, ExplicitPolicy, FormatterMixin

__all__ = ("CompoundBaseProblem", "SingleCompoundProblem", "MultiCompoundProblem", "CompoundProblem")

from moscot.solvers._tagged_array import Tag, TaggedArray

Callback_t = Optional[
    Union[
        Literal["pca_local"],
        Callable[[AnnData, Optional[AnnData], ProblemKind, Any], Tuple[TaggedArray, Optional[TaggedArray]]],
    ]
]


B = TypeVar("B", bound=BaseProblem)
K = TypeVar("K")  # TODO(michalk8): finish me


class CompoundBaseProblem(BaseProblem, Generic[K, B], ABC):
    def __init__(self, adata: AnnData):
        super().__init__(adata)
        self._problems: Optional[Dict[Tuple[K, K], B]] = None
        self._solutions: Optional[Dict[Tuple[K, K], BaseSolverOutput]] = None
        self._policy: Optional[SubsetPolicy] = None

    @abstractmethod
    def _create_problem(
        self, src: K, tgt: K, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any
    ) -> OTProblem:
        pass

    @abstractmethod
    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> SubsetPolicy:
        pass

    @property
    @abstractmethod
    def _base_problem_type(self) -> Type[B]:
        pass

    @property
    @abstractmethod
    def _valid_policies(self) -> Tuple[str, ...]:
        pass

    # TODO(michalk8): refactor me
    def _create_problems(
        self,
        callback: Callback_t = None,
        callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> Dict[Tuple[K, K], B]:
        problems = {}
        for (src, tgt), (src_mask, tgt_mask) in self._policy.create_masks().items():
            kwargs_ = dict(kwargs)
            if isinstance(self._policy, FormatterMixin):
                src = self._policy._format(src, is_source=True)
                tgt = self._policy._format(tgt, is_source=False)
            problem = self._create_problem(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            # TODO(michalk8): refactor me
            if callback is not None:
                callback = problem._prepare_callback if callback == "pca_local" else callback
                x, y = callback(
                    problem.adata,
                    problem._adata_y,
                    problem._problem_kind,
                    **callback_kwargs,
                )
                if problem._problem_kind != ProblemKind.QUAD_FUSED:
                    kwargs_["xy"] = (x, y)
                elif x is not None and y is not None:
                    kwargs_["x"] = x
                    kwargs_["y"] = y

            problems[src, tgt] = problem.prepare(**kwargs_)

        return problems

    def prepare(
        self,
        key: str,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        subset: Optional[Sequence[Tuple[Any, Any]]] = None,
        reference: Optional[Any] = None,
        axis: Axis_t = "obs",
        callback: Optional[Union[Literal["pca_local"], Callable[[AnnData, Any], npt.ArrayLike]]] = None,
        callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "CompoundProblem":
        if self._valid_policies and policy not in self._valid_policies:
            raise ValueError(f"TODO: Invalid policy `{policy}`")

        self._policy = self._create_policy(policy=policy, key=key, axis=axis)
        if isinstance(self._policy, ExplicitPolicy):
            self._policy = self._policy(subset)
        elif isinstance(self._policy, StarPolicy):
            self._policy = self._policy(filter=subset, reference=reference)
        else:
            self._policy = self._policy(filter=subset)

        self._problems = self._create_problems(callback=callback, callback_kwargs=callback_kwargs, **kwargs)
        self._solutions = None

        return self

    # TODO(michalk8): remove OT specific arguments
    def solve(
        self,
        epsilon: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "CompoundProblem":
        self._solutions = {}

        for subset, problem in self.problems.items():
            self.solutions[subset] = problem.solve(
                epsilon=epsilon, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs
            ).solution

        return self

    # TODO(michalk8): simpliy/split
    def _apply(
        self,
        data: Optional[Union[str, npt.ArrayLike, Mapping[Tuple[Any, Any], Union[str, npt.ArrayLike]]]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
        forward: bool = True,
        return_all: bool = False,
        scale_by_marginals: bool = False,
        **kwargs: Any,
    ) -> Union[Dict[Tuple[Any, Any], npt.ArrayLike], Dict[Tuple[Any, Any], Dict[Tuple[Any, Any], npt.ArrayLike]]]:
        def get_data(plan: Tuple[Any, Any]) -> Optional[npt.ArrayLike]:
            if isinstance(data, np.ndarray):
                return data
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
            if forward:
                initial_problem = self.problems[steps[0]]
                current_mass = initial_problem._get_mass(
                    initial_problem.adata, data=get_data(plan), subset=subset, normalize=normalize
                )
            else:
                steps = steps[::-1]
                initial_problem = self.problems[steps[0]]
                current_mass = initial_problem._get_mass(
                    initial_problem.adata if initial_problem._adata_y is None else initial_problem._adata_y,
                    data=get_data(plan),
                    subset=subset,
                    normalize=normalize,
                )

            ds = {}  # TODO(michalk8)
            ds[steps[0][0] if forward else steps[0][1]] = current_mass
            for step in steps:
                if step not in self.problems:
                    raise ValueError(f"No transport map computed for {step}")
                problem = self.problems[step]
                fun = problem.push if forward else problem.pull
                current_mass = fun(
                    current_mass, subset=subset, normalize=normalize, scale_by_marginals=scale_by_marginals
                )
                ds[step[1] if forward else step[0]] = current_mass

            res[plan] = ds if return_all else current_mass
        # TODO(michalk8): return the values iff only 1 plan?
        return res

    def _extract_cost_matrix(self, key: str, *, src: Any, tgt: Any) -> TaggedArray:
        if self._policy is None:
            raise ValueError("TODO: no policy initialized")

        attr = f"{self._policy.axis}p"
        try:
            data = getattr(self.adata, attr)[key]
        except KeyError:
            raise KeyError(f"TODO: data not in `adata.{attr}[{key!r}]`") from None

        src_mask = self._policy.create_mask(src, allow_empty=False)
        tgt_mask = self._policy.create_mask(tgt, allow_empty=False)
        return TaggedArray(data[src_mask, :][:, tgt_mask], tag=Tag.COST_MATRIX)

    def push(self, *args: Any, **kwargs: Any) -> Union[npt.ArrayLike, Dict[Any, npt.ArrayLike]]:
        return self._apply(*args, forward=True, **kwargs)

    def pull(self, *args: Any, **kwargs: Any) -> Union[npt.ArrayLike, Dict[Any, npt.ArrayLike]]:
        return self._apply(*args, forward=False, **kwargs)

    @property
    def _default_solver(self) -> OTSolver:
        return SinkhornSolver()

    @property
    def problems(self) -> Optional[Dict[Tuple[Any, Any], B]]:
        return self._problems

    @property
    def solutions(self) -> Optional[Dict[Tuple[Any, Any], BaseSolverOutput]]:
        return self._solutions

    def __getitem__(self, item: Tuple[Any, Any]) -> B:
        return self.problems[item]

    def __len__(self) -> int:
        return 0 if self.problems is None else len(self.problems)

    def __iter__(self) -> Iterator:
        if self.problems is None:
            raise StopIteration
        return iter(self.problems)


class SingleCompoundProblem(CompoundBaseProblem, ABC):
    def _create_problem(self, src: Any, tgt: Any, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any) -> B:
        return self._base_problem_type(
            self._mask(src_mask),
            self._mask(tgt_mask),
            source=src,
            target=tgt,
            **kwargs,
        )

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit", "external_star"] = "sequential",
        key: Optional[str] = None,
        axis: Axis_t = "obs",
        **_: Any,
    ) -> SubsetPolicy:
        return (
            SubsetPolicy.create(policy, self.adata, key=key, axis=axis)
            if isinstance(policy, str)
            else ExplicitPolicy(self.adata, key=key, axis=axis)
        )

    def _mask(self, mask: npt.ArrayLike) -> AnnData:
        # TODO(michalk8): can include logging/extra sanity that mask is not empty
        return self.adata[mask] if self._policy.axis == "obs" else self.adata[:, mask]


class MultiCompoundProblem(CompoundBaseProblem, ABC):
    _KEY = "subset"

    def __init__(
        self,
        *adatas: Union[AnnData, Mapping[Any, AnnData], Tuple[AnnData, ...], List[AnnData]],
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
        super().__init__(adata)

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
        kwargs["axis"] = "obs"
        return super().prepare(None, subset=subset, policy=policy, reference=reference, **kwargs)

    def _create_problem(self, src: Any, tgt: Any, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any) -> B:
        return self._base_problem_type(self._adatas[src], self._adatas[tgt], source=src, target=tgt, **kwargs)

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **_: Any,
    ) -> SubsetPolicy:
        return (
            SubsetPolicy.create(policy, self._policy_adata, key=self._KEY, axis="obs")
            if isinstance(policy, str)
            else ExplicitPolicy(self._policy_adata, key=self._KEY, axis="obs")
        )


class CompoundProblem(CompoundBaseProblem, ABC):
    def __init__(self, *adatas: Union[AnnData, Mapping[Any, AnnData], Tuple[AnnData, ...], List[AnnData]]):
        if len(adatas) == 1 and isinstance(adatas[0], AnnData):
            self._prob = SingleCompoundProblem(adatas[0])
        else:
            self._prob = MultiCompoundProblem(*adatas)

        super().__init__(self._prob.adata)

    def _create_problem(self, *args: Any, **kwargs: Any) -> Dict[Tuple[Any, Any], B]:
        return self._prob._create_problem(*args, **kwargs)

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        key: Optional[str] = None,
        **_: Any,
    ) -> SubsetPolicy:
        self._prob._policy = self._prob._create_policy(policy=policy, key=key)
        return self._prob._policy

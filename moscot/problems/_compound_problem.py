from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Dict, List, Type, Tuple, Union, Literal, Mapping, Callable, Iterator, Optional, Sequence

import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot._docs import d
from moscot.backends.ott import SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.problems._base_problem import BaseProblem, GeneralProblem
from moscot.problems._subset_policy import Axis_t, StarPolicy, SubsetPolicy, ExplicitPolicy, FormatterMixin

__all__ = ("SingleCompoundProblem", "MultiCompoundProblem", "CompoundProblem")

from moscot.solvers._tagged_array import TaggedArray

Callback_t = Optional[
    Union[
        Literal["pca_local"],
        Callable[[AnnData, Optional[AnnData], ProblemKind, Any], Tuple[TaggedArray, Optional[TaggedArray]]],
    ]
]

@d.get_sections(base="CompoundBaseProblem", sections=["Parameters", "Raises"])
@d.dedent
class CompoundBaseProblem(BaseProblem, ABC):
    """
    Base class for all biological problems.

    Base class translating a biological problem to potentially multiple Optimal Transport problems.

    Parameters
    ----------
    adata
        instance of :class:`anndata.AnnData` containing the data defining the biological problem.
    solver
        instance of :class:`moscot.solvers` used for solving the optimal transport problem(s)
    base_problem_type
        subclass of :class:`moscot.problems.GeneralProblem` defining the problem type of a single optimal transport problem


    Raises
    ------
    TypeError
        If ``base_problem_type`` is not a subclass of `GeneralProblem`

    """
    def __init__(
        self,
        adata: AnnData,
        solver: Optional[BaseSolver] = None,
        *,
        # TODO(michalk8): properly type this
        base_problem_type: Type[GeneralProblem] = GeneralProblem,
    ):
        super().__init__(adata, solver=solver)

        self._problems: Optional[Dict[Tuple[Any, Any], GeneralProblem]] = None
        self._solutions: Optional[Dict[Tuple[Any, Any], BaseSolverOutput]] = None
        self._policy: Optional[SubsetPolicy] = None
        if not issubclass(base_problem_type, GeneralProblem):
            raise TypeError("TODO: `base_problem_type` must be a subtype of `GeneralProblem`.")
        self._base_problem_type = base_problem_type

    @abstractmethod
    def _create_problem(
        self, src: Any, tgt: Any, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any
    ) -> GeneralProblem:
        pass

    @abstractmethod
    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> SubsetPolicy:
        pass

    @d.dedent
    def _create_problems(
        self,
        callback: Callback_t = None,
        callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> Dict[Tuple[Any, Any], GeneralProblem]:
        problems = {}
        for (src, tgt), (src_mask, tgt_mask) in self._policy.mask().items():
            kwargs_ = dict(kwargs)
            if isinstance(self._policy, FormatterMixin):
                src = self._policy._format(src, is_source=True)
                tgt = self._policy._format(tgt, is_source=False)
            problem = self._create_problem(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            if callback is not None:
                callback = problem._prepare_callback if callback == "pca_local" else callback
                x, y = callback(
                    problem.adata,
                    problem._adata_y,
                    problem.solver.problem_kind,
                    **callback_kwargs,
                )
                if problem.solver.problem_kind != ProblemKind.QUAD_FUSED:
                    kwargs_["x"] = x
                    kwargs_["y"] = y
                elif x is not None and y is not None:
                    kwargs_["xy"] = (x, y)

            problems[src, tgt] = problem.prepare(**kwargs_)

        return problems

    @d.get_sections(base="CompoundBaseProblem_prepare", sections=["Parameters", "Raises"])
    @d.dedent
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
        """
        Prepares the biological problem

        Parameters
        ----------
        key
            key in :attr:`anndata.AnnData.obs` allocating the cell to a certain cell distribution
        policy
            defines which transport maps to compute given different cell distributions
        subset
            subset of `anndata.AnnData.obs` [key] values of which the policy is to be applied to
        reference
            pass
        axis
            pass
        callback
            pass
        callback_kwargs
            pass
        kwargs
            pass

        Returns
        -------
        Self

        """
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

    @d.dedent
    def solve(
        self,
        epsilon: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        **kwargs: Any,
    ) -> "CompoundProblem":
        self._solutions = {}
        for subset, problem in self.problems.items():
            self.solutions[subset] = problem.solve(
                epsilon=epsilon, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs
            ).solution

        return self

    @d.get_sections(base="_apply", sections=["Parameters", "Raises"])
    @d.dedent
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
        """
        Base function to use a transport map(s) as linear operator.

        Parameters
        ----------

        data
            If `data` is of type `str` this should correspond to a column in :attr:`anndata.AnnData.obs`. The transport map is applied to the subset corresponding to the source distribution (if `forward` is `True`) or target distribution (if `forward` is `False`) of that column.
            If `data` is of type :class:npt.ArrayLike the transport map is applied to `data`
            If `data` is a mapping then the keys should correspond to the tuple defining a single optimal transport map and the value should be one of the two cases described above
        subset
            If `data` is a column in :attr:`anndata.AnnData.obs` the distribution the transport map is applied to only has mass on those cells which are in `subset` when filtering for :attr:`anndata.AnnData.obs`
        normalize
            Whether to normalize the result to 1 after the transport map has been applied
        forward
            If `True` the data is pushed from the source to the target distribution. If `False` the mass is pulled from the target distribution to the source distribution
        return_all
            If `True` and transport maps are applied consecutively only the final mass is returned. Otherwise, all intermediate step results are returned, too
        scale_by_marginals
            If `True` the transport map is scaled to be a stochastic matrix by multiplying the resulting mass by the inverse of the marginals, EXAMPLE

        Raises
        ------
        ValueError
            If a transport map between the corresponding source and target distribution is not computed
        """

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

            ds = {}
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

    def push(self, *args: Any, **kwargs: Any) -> Union[npt.ArrayLike, Dict[Any, npt.ArrayLike]]:
        return self._apply(*args, forward=True, **kwargs)

    def pull(self, *args: Any, **kwargs: Any) -> Union[npt.ArrayLike, Dict[Any, npt.ArrayLike]]:
        return self._apply(*args, forward=False, **kwargs)

    @property
    def _default_solver(self) -> BaseSolver:
        return SinkhornSolver()

    @property
    def problems(self) -> Optional[Dict[Tuple[Any, Any], GeneralProblem]]:
        return self._problems

    @property
    def solutions(self) -> Optional[Dict[Tuple[Any, Any], BaseSolverOutput]]:
        return self._solutions

    def __getitem__(self, item: Tuple[Any, Any]) -> GeneralProblem:
        return self.problems[item]

    def __len__(self) -> int:
        return 0 if self.problems is None else len(self.problems)

    def __iter__(self) -> Iterator:
        if self.problems is None:
            raise StopIteration
        return iter(self.problems)


class SingleCompoundProblem(CompoundBaseProblem):
    def _create_problem(
        self, src: Any, tgt: Any, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any
    ) -> GeneralProblem:
        return self._base_problem_type(
            self._mask(src_mask),
            self._mask(tgt_mask),
            source=src,
            target=tgt,
            solver=self.solver,
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

    # TODO(MUCKD): this should not be here
    def _dict_to_adata(self, d: Mapping[str, npt.ArrayLike], obs_key: str) -> None:
        tmp = np.empty(len(self.adata))
        tmp[:] = np.nan
        for key, value in d.items():
            mask = self.adata.obs[self._temporal_key] == key
            tmp[mask] = np.squeeze(value)
        self.adata.obs[obs_key] = tmp


class MultiCompoundProblem(CompoundBaseProblem):
    _KEY = "subset"

    def __init__(
        self,
        *adatas: Union[AnnData, Mapping[Any, AnnData], Tuple[AnnData, ...], List[AnnData]],
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
        kwargs["axis"] = "obs"
        return super().prepare(None, subset=subset, policy=policy, reference=reference, **kwargs)

    def _create_problem(
        self, src: Any, tgt: Any, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any
    ) -> GeneralProblem:
        return self._base_problem_type(
            self._adatas[src], self._adatas[tgt], source=src, target=tgt, solver=self._solver, **kwargs
        )

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


class CompoundProblem(CompoundBaseProblem):
    def __init__(
        self,
        *adatas: Union[AnnData, Mapping[Any, AnnData], Tuple[AnnData, ...], List[AnnData]],
        solver: Optional[BaseSolver] = None,
        **kwargs: Any,
    ):
        if len(adatas) == 1 and isinstance(adatas[0], AnnData):
            self._prob = SingleCompoundProblem(adatas[0], solver=solver, **kwargs)
        else:
            self._prob = MultiCompoundProblem(*adatas, solver=solver, **kwargs)

        super().__init__(self._prob.adata, solver=self._prob.solver)

    def _create_problem(self, *args: Any, **kwargs: Any) -> Dict[Tuple[Any, Any], GeneralProblem]:
        return self._prob._create_problem(*args, **kwargs)

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        key: Optional[str] = None,
        **_: Any,
    ) -> SubsetPolicy:
        self._prob._policy = self._prob._create_policy(policy=policy, key=key)
        return self._prob._policy

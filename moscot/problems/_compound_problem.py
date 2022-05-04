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
    Hashable,
    Iterator,
    Optional,
    Sequence,
)

from scipy.sparse import csr_matrix
import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot._docs import d
from moscot.solvers._output import BaseSolverOutput
from moscot.problems._base_problem import OTProblem, BaseProblem
from moscot.problems._subset_policy import Axis_t, StarPolicy, SubsetPolicy, ExplicitPolicy, FormatterMixin

__all__ = ("CompoundBaseProblem", "SingleCompoundProblem", "MultiCompoundProblem")

from moscot.solvers._tagged_array import Tag, TaggedArray

B = TypeVar("B", bound=OTProblem)
K = TypeVar("K", bound=Hashable)
Key = Tuple[K, K]
Callback_t = Callable[[AnnData, AnnData, Any], Mapping[str, TaggedArray]]


@d.get_sections(base="CompoundBaseProblem", sections=["Parameters", "Raises"])
@d.dedent
class CompoundBaseProblem(BaseProblem, Generic[K, B], ABC):
    """
    Base class for all biological problems.

    This base class translates a biological problem to potentially multiple Optimal Transport problems.

    Parameters
    ----------
    %(adata)s

    Raises
    ------
    TypeError
        If `base_problem_type` is not a subclass of `GeneralProblem`.
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)
        self._problems: Optional[Dict[Key, B]] = None
        self._solutions: Optional[Dict[Key, BaseSolverOutput]] = None
        self._policy: Optional[SubsetPolicy] = None

    @abstractmethod
    def _create_problem(self, src: K, tgt: K, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any) -> B:
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

    def _callback_handler(
        self,
        src: K,
        tgt: K,
        problem: B,
        callback: Union[Literal["local-pca"], Callback_t],
        callback_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if callback == "local-pca":
            callback = problem._local_pca_callback
        if not callable(callback):
            raise TypeError("TODO: callback not callable")

        # TODO(michalk8): consider passing `adata` that only has `src`/`tgt`
        data = callback(problem.adata, problem._adata_y, **callback_kwargs)
        if not isinstance(data, Mapping):
            raise TypeError("TODO: callback did not return a mapping.")
        return data

    def _create_problems(
        self,
        callback: Optional[Union[str, Callback_t]] = None,
        callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> Dict[Key, B]:
        problems = {}
        for (src, tgt), (src_mask, tgt_mask) in self._policy.create_masks().items():
            if isinstance(self._policy, FormatterMixin):
                src_name = self._policy._format(src, is_source=True)
                tgt_name = self._policy._format(tgt, is_source=False)
            else:
                src_name = src
                tgt_name = tgt

            problem = self._create_problem(src=src_name, tgt=tgt_name, src_mask=src_mask, tgt_mask=tgt_mask)
            if callback is not None:
                data = self._callback_handler(src, tgt, problem, callback, callback_kwargs=callback_kwargs)
                kws = {**kwargs, **data}
            else:
                kws = kwargs

            problems[src_name, tgt_name] = problem.prepare(**kws)

        return problems

    @d.get_sections(base="CompoundBaseProblem_prepare", sections=["Parameters", "Raises"])
    @d.dedent
    def prepare(
        self,
        key: str,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        subset: Optional[Sequence[Key]] = None,
        reference: Optional[Any] = None,
        axis: Axis_t = "obs",
        callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "CompoundBaseProblem":
        """
        Prepare the biological problem.

        Parameters
        ----------
        key
            key in :attr:`anndata.AnnData.obs` allocating the cell to a certain cell distribution
        policy
            defines which transport maps to compute given different cell distributions
        subset
            subset of `anndata.AnnData.obs` [key] values of which the policy is to be applied to
        %(reference)s
        %(axis)s
        %(callback)s
        %(callback_kwargs)s
        kwargs
            keyword arguments for

        Returns
        -------
        :class:moscot.problems.CompoundProblem

        Raises
        ------
        TODO.
        """
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
        for p in self.problems.values():
            self._problem_kind = p._problem_kind
            break

        return self

    def solve(self, *args: Any, **kwargs: Any) -> "CompoundBaseProblem":
        """
        Solve the biological problem.

        Parameters
        ----------
        args
            TODO.
        kwargs
            Keyword arguments for one of
                - :attr:`moscot.problems.GeneralProblem.solve()`
                - :attr:`moscot.problems.MultiMarginalProblem.solve()`
                - :attr:`moscot.problems.TemporalBaseProblem.solve()`

        Raises
        ------
        """
        if self._problem_kind is None:
            raise RuntimeError("Run .prepare() first")

        self._solutions = {}
        for subset, problem in self.problems.items():
            self.solutions[subset] = problem.solve(*args, **kwargs).solution

        return self

    @d.get_sections(base="_apply", sections=["Parameters", "Raises"])
    @d.dedent
    def _apply(
        self,
        data: Optional[Union[str, npt.ArrayLike, Mapping[Key, Union[str, npt.ArrayLike]]]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
        forward: bool = True,
        return_all: bool = False,
        scale_by_marginals: bool = False,
        **kwargs: Any,
    ) -> Union[Dict[Tuple[Any, Any], npt.ArrayLike], Dict[Tuple[Any, Any], Dict[Tuple[Any, Any], npt.ArrayLike]]]:
        """
        Use (a) transport map(s) as a linear operator.

        Parameters
        ----------
        %(data)s
        %(subset)s
        %(normalize)s
        forward
            If `True` the data is pushed from the source to the target distribution. If `False` the mass is pulled
            from the target distribution to the source distribution.
        return_all
            If `True` and transport maps are applied consecutively only the final mass is returned. Otherwise,
            all intermediate step results are returned, too.
        %(scale_by_marginals)s

        Returns
        -------
        TODO.

        Raises
        ------
        ValueError
            If a transport map between the corresponding source and target distribution is not computed
        """

        def get_data(plan: Key) -> Optional[npt.ArrayLike]:
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
        res: Dict[Key, npt.ArrayLike] = {}
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

    @d.get_sections(base="CompoundBaseProblem_push", sections=["Parameters", "Raises"])
    @d.dedent
    def push(self, *args: Any, **kwargs: Any) -> Union[npt.ArrayLike, Dict[Any, npt.ArrayLike]]:
        """
        Push mass from `start` to `end`. TODO: verify.

        Parameters
        ----------
        %(data)s
        %(subset)s
        %(normalize)s
        return_all
            If `True` and transport maps are applied consecutively only the final mass is returned. Otherwise,
            all intermediate step results are returned, too.
        %(scale_by_marginals)s
        kwargs
            keyword arguments for :meth:`moscot.problems.CompoundProblem._apply()`

        Returns
        -------
        TODO.

        Raises
        ------
        %(_apply.raises)s
        """
        return self._apply(*args, forward=True, **kwargs)

    @d.get_sections(base="CompoundBaseProblem_pull", sections=["Parameters", "Raises"])
    @d.dedent
    def pull(self, *args: Any, **kwargs: Any) -> Union[npt.ArrayLike, Dict[Any, npt.ArrayLike]]:
        """
        Pull mass from `end` to `start`. TODO: expose kwargs.

        Parameters
        ----------
        %(data)s
        %(subset)s
        %(normalize)s
        return_all
            If `True` and transport maps are applied consecutively only the final mass is returned. Otherwise,
            all intermediate step results are returned, too.
        %(scale_by_marginals)s
        kwargs
            keyword arguments for :meth:`moscot.problems.CompoundProblem._apply()`

        Returns
        -------
        TODO.

        Raises
        ------
        %(_apply.raises)s
        """
        return self._apply(*args, forward=False, **kwargs)

    @property
    def problems(self) -> Optional[Dict[Key, B]]:
        return self._problems

    @property
    def solutions(self) -> Optional[Dict[Key, BaseSolverOutput]]:
        return self._solutions

    def __getitem__(self, item: Key) -> B:
        return self.problems[item]

    def __len__(self) -> int:
        return 0 if self.problems is None else len(self.problems)

    def __iter__(self) -> Iterator[Key]:
        if self.problems is None:
            raise StopIteration
        return iter(self.problems)


@d.get_sections(base="SingleCompoundProblem", sections=["Parameters", "Raises"])
@d.dedent
class SingleCompoundProblem(CompoundBaseProblem, Generic[K, B], ABC):
    """
    Class handling biological problems composed of exactly one :class:`anndata.AnnData` instance.

    This class is needed to apply the `policy` to one :class:`anndata.AnnData` objects and hence create the
    Optimal Transport subproblems from the biological problem.

    Parameters
    ----------
    %(CompoundBaseProblem.parameters)s

    Raises
    ------
    %(CompoundBaseProblem.raises)s
    """

    def _create_problem(self, src: K, tgt: K, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any) -> B:
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
        return self.adata[mask] if self._policy.axis == "obs" else self.adata[:, mask]

    def _callback_handler(
        self,
        src: K,
        tgt: K,
        problem: B,
        callback: Union[Literal["local-pca", "cost-matrix"], Callback_t],
        callback_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        # TODO(michalk8): better name?
        if callback == "cost-matrix":
            return self._cost_matrix_callback(src, tgt, **callback_kwargs)

        return super()._callback_handler(src, tgt, problem, callback, callback_kwargs=callback_kwargs)

    def _cost_matrix_callback(
        self, src: K, tgt: K, *, key: str, return_linear: bool = True, **_: Any
    ) -> Dict[Literal["xy", "x", "y"], TaggedArray]:
        attr = f"{self._policy.axis}p"
        try:
            data = getattr(self.adata, attr)[key]
        except KeyError:
            raise KeyError(f"TODO: data not in `adata.{attr}[{key!r}]`") from None

        src_mask = self._policy.create_mask(src, allow_empty=False)
        tgt_mask = self._policy.create_mask(tgt, allow_empty=False)

        if return_linear:
            return {"xy": TaggedArray(data[src_mask, :][:, tgt_mask], tag=Tag.COST_MATRIX)}

        return {
            "x": TaggedArray(data[src_mask, :][:, src_mask], tag=Tag.COST_MATRIX),
            "y": TaggedArray(data[tgt_mask, :][:, tgt_mask], tag=Tag.COST_MATRIX),
        }


@d.get_sections(base="MultiCompoundProblem", sections=["Parameters", "Raises"])
@d.dedent
class MultiCompoundProblem(CompoundBaseProblem, Generic[K, B], ABC):
    """
    Class handling biological problems composed of more than one :class:`anndata.AnnData` instance.

    This class is needed to apply the `policy` to multiple :class:`anndata.AnnData` objects and hence create the
    Optimal Transport subproblems from the biological problem.

    Parameters
    ----------
    %(adatas)s
    %(solver)s
    kwargs
        keyword arguments for :class:`moscot.problems.CompoundBaseProblem`

    Raises
    ----------
    %(CompoundBaseProblem.raises)s
    """

    _SUBSET_KEY = "subset"

    def __init__(
        self,
        *adatas: Union[AnnData, Mapping[K, AnnData], Tuple[AnnData, ...], List[AnnData]],
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
        super().__init__(adata, **kwargs)

        if not isinstance(adatas, Mapping):
            adatas = {i: adata for i, adata in enumerate(adatas)}

        self._adatas: Mapping[K, AnnData] = adatas
        self._policy_adata = AnnData(
            csr_matrix((len(self._adatas), 1), dtype=float),
            obs=pd.Series(list(self._adatas.keys()), dtype="category").to_frame(self._SUBSET_KEY),
            dtype=float,
        )

    def prepare(
        self,
        subset: Optional[Sequence[K]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        reference: Optional[Any] = None,
        **kwargs: Any,
    ) -> "MultiCompoundProblem":
        kwargs["axis"] = "obs"
        return super().prepare(None, subset=subset, policy=policy, reference=reference, **kwargs)

    def _create_problem(self, src: K, tgt: K, src_mask: npt.ArrayLike, tgt_mask: npt.ArrayLike, **kwargs: Any) -> B:
        return self._base_problem_type(self._adatas[src], self._adatas[tgt], source=src, target=tgt, **kwargs)

    def _create_policy(
        self,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **_: Any,
    ) -> SubsetPolicy:
        return (
            SubsetPolicy.create(policy, self._policy_adata, key=self._SUBSET_KEY, axis="obs")
            if isinstance(policy, str)
            else ExplicitPolicy(self._policy_adata, key=self._SUBSET_KEY, axis="obs")
        )

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Type,
    Tuple,
    Union,
    Generic,
    Mapping,
    TypeVar,
    Callable,
    Hashable,
    Iterator,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

from scipy.sparse import issparse
from typing_extensions import Literal

from anndata import AnnData

from moscot._docs import d
from moscot._types import ArrayLike
from moscot.problems._utils import require_prepare
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.base._utils import attributedispatch
from moscot._constants._constants import Policy
from moscot.solvers._tagged_array import Tag, TaggedArray
from moscot.problems._subset_policy import (
    Axis_t,
    StarPolicy,
    DummyPolicy,
    SubsetPolicy,
    OrderedPolicy,
    ExplicitPolicy,
    FormatterMixin,
)
from moscot.problems.base._base_problem import OTProblem, BaseProblem, ProblemStage
from moscot.problems.base._problem_manager import ProblemManager

__all__ = ["BaseCompoundProblem", "CompoundProblem"]

K = TypeVar("K", bound=Hashable)
B = TypeVar("B", bound=OTProblem)
Callback_t = Callable[[AnnData, AnnData], Mapping[str, TaggedArray]]
ApplyOutput_t = Union[ArrayLike, Dict[K, ArrayLike]]
# TODO(michalk8): future behavior
# ApplyOutput_t = Union[ArrayLike, Dict[Tuple[K, K], ArrayLike]]


@d.get_sections(base="BaseCompoundProblem", sections=["Parameters", "Raises"])
@d.dedent
class BaseCompoundProblem(BaseProblem, ABC, Generic[K, B]):
    """
    Base class for all biological problems.

    This base class translates a biological problem to potentially multiple Optimal Transport problems.

    Parameters
    ----------
    %(adata)s

    Raises
    ------
    TypeError
        If `base_problem_type` is not a subclass of :class:`moscot.problems.OTProblem`.
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)
        self._problem_manager: Optional[ProblemManager[K, B]] = None

    @abstractmethod
    def _create_problem(self, src_mask: ArrayLike, tgt_mask: ArrayLike, **kwargs: Any) -> B:
        pass

    @abstractmethod
    def _create_policy(
        self,
        policy: Policy,
        **kwargs: Any,
    ) -> SubsetPolicy[K]:
        pass

    @property
    @abstractmethod
    def _valid_policies(self) -> Tuple[str, ...]:
        pass

    # TODO(michalk8): refactor me
    def _callback_handler(
        self,
        src: K,
        tgt: K,
        problem: B,
        callback: Union[Literal["local-pca"], Callback_t],
        callback_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if callback == "local-pca":
            callback = problem._local_pca_callback  # type: ignore[assignment]
        if not callable(callback):
            raise TypeError("TODO: callback not callable")

        # TODO(michalk8): consider passing `adata` that only has `src`/`tgt`
        data = callback(problem.adata, problem._adata_y, **callback_kwargs)
        if not isinstance(data, Mapping):
            raise TypeError("TODO: callback did not return a mapping.")
        return data

    # TODO(michalk8): refactor me
    def _create_problems(
        self,
        callback: Optional[Union[str, Callback_t]] = None,
        callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> Dict[Tuple[K, K], B]:
        from moscot.problems.base._birth_death import BirthDeathProblem

        if TYPE_CHECKING:
            assert isinstance(self._policy, SubsetPolicy)

        problems: Dict[Tuple[K, K], B] = {}
        for (src, tgt), (src_mask, tgt_mask) in self._policy.create_masks().items():
            if isinstance(self._policy, FormatterMixin):
                src_name = self._policy._format(src, is_source=True)
                tgt_name = self._policy._format(tgt, is_source=False)
            else:
                src_name = src
                tgt_name = tgt

            problem = self._create_problem(src_mask=src_mask, tgt_mask=tgt_mask)
            if callback is not None:
                data = self._callback_handler(
                    src, tgt, problem, callback, callback_kwargs=callback_kwargs  # type: ignore[arg-type]
                )
                kws = {**kwargs, **data}
            else:
                kws = kwargs

            if isinstance(problem, BirthDeathProblem):
                kws["delta"] = tgt - src  # type: ignore[operator]
                kws["proliferation_key"] = self.proliferation_key  # type: ignore[attr-defined]
                kws["apoptosis_key"] = self.apoptosis_key  # type: ignore[attr-defined]
            problems[src_name, tgt_name] = problem.prepare(**kws)

        return problems

    @d.get_sections(base="BaseCompoundProblem_prepare", sections=["Parameters", "Raises"])
    @d.dedent
    def prepare(
        self,
        key: str,
        policy: Policy = Policy.SEQUENTIAL,
        subset: Optional[Sequence[Tuple[K, K]]] = None,
        reference: Optional[Any] = None,
        axis: Axis_t = "obs",
        callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "BaseCompoundProblem[K,B]":
        """
        Prepare the biological problem.

        Parameters
        ----------
        %(key)s
        policy
            Defines which transport maps to compute given different cell distributions.
        subset
            Subset of :attr:`anndata.AnnData.obs` ``['{key}']`` values of which the policy is to be applied to.
        %(reference)s
        %(axis)s
        %(callback)s
        %(callback_kwargs)s
        %(a)s
        %(b)s
        kwargs
            keyword arguments for something.

        Returns
        -------
        :class:`moscot.problems.CompoundProblem`.
        """
        if self._valid_policies and policy not in self._valid_policies:
            raise ValueError(f"TODO: Invalid policy `{policy}`")

        policy = self._create_policy(policy=policy, key=key, axis=axis)  # type: ignore[assignment]
        if TYPE_CHECKING:
            assert isinstance(policy, SubsetPolicy)
        if isinstance(policy, ExplicitPolicy):
            policy = policy(subset=subset)
        elif isinstance(policy, StarPolicy):
            policy = policy(reference=reference)
        else:
            policy = policy()  # type: ignore[assignment]

        # TODO(michalk8): manager must be currently instantiated first, since `_create_problems` accesses the policy
        # when refactoring the callback, consider changing this
        self._problem_manager = ProblemManager(self, policy=policy)
        problems = self._create_problems(callback=callback, callback_kwargs=callback_kwargs, **kwargs)
        self._problem_manager.add_problems(problems)

        for p in self.problems.values():
            self._problem_kind = p._problem_kind
            break
        return self

    def solve(self, stage: Union[ProblemStage, Tuple[ProblemStage, ...]] = (ProblemStage.PREPARED, ProblemStage.SOLVED), **kwargs: Any) -> "BaseCompoundProblem[K,B]":  # type: ignore[override] # noqa: E501
        """
        Solve the biological problem.

        Parameters
        ----------
        stage
            Some stage TODO.
        kwargs
            Keyword arguments for one of:
                - :meth:`moscot.problems.OTProblem.solve`.
                - :meth:`moscot.problems.MultiMarginalProblem.solve`.
                - :meth:`moscot.problems.BirthDeathProblem.solve`.

        Returns
        -------
        :class:`moscot.problems.CompoundProblem`.
        """
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        problems = self._problem_manager.get_problems(stage=stage)
        # TODO(michalk8): print how many problems are being solved?
        for _, problem in problems.items():
            _ = problem.solve(**kwargs)

        return self

    @attributedispatch(attr="_policy")
    def _apply(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        forward: bool = True,
        scale_by_marginals: bool = False,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        raise NotImplementedError(type(self._policy))

    @_apply.register(DummyPolicy)
    @_apply.register(StarPolicy)
    def _(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        forward: bool = True,
        scale_by_marginals: bool = False,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        if TYPE_CHECKING:
            assert isinstance(self._policy, StarPolicy)

        res = {}
        # TODO(michalk8): should use manager.plan (once implemented), as some problems may not be solved
        for src, tgt in self._policy.plan(explicit_steps=kwargs.pop("explicit_steps", None)):
            problem = self.problems[src, tgt]
            fun = problem.push if forward else problem.pull
            res[src] = fun(data=data, scale_by_marginals=scale_by_marginals, **kwargs)
        return res

    @_apply.register(ExplicitPolicy)
    @_apply.register(OrderedPolicy)
    def _(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        forward: bool = True,
        scale_by_marginals: bool = False,
        start: Optional[K] = None,
        end: Optional[K] = None,
        return_all: bool = False,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        explicit_steps = kwargs.pop(
            "explicit_steps", [[start, end]] if isinstance(self._policy, ExplicitPolicy) else None
        )
        if TYPE_CHECKING:
            assert isinstance(self._policy, OrderedPolicy)
        (src, tgt), *rest = self._policy.plan(
            forward=forward,
            start=start,
            end=end,
            explicit_steps=explicit_steps,
        )
        problem = self.problems[src, tgt]
        adata = problem.adata if forward else problem._adata_y
        current_mass = problem._get_mass(adata, data=data, **kwargs)
        # TODO(michlak8): future behavior
        # res = {(None, src) if forward else (tgt, None): current_mass}
        res = {src if forward else tgt: current_mass}
        for _src, _tgt in [(src, tgt)] + rest:
            problem = self.problems[_src, _tgt]
            fun = problem.push if forward else problem.pull
            res[_tgt] = current_mass = fun(current_mass, scale_by_marginals=scale_by_marginals, **kwargs)

        return res if return_all else current_mass

    @d.get_sections(base="BaseCompoundProblem_push", sections=["Parameters", "Raises"])
    @d.dedent
    def push(self, *args: Any, **kwargs: Any) -> ApplyOutput_t[K]:
        """
        Push mass from `start` to `end`.

        TODO: verify.

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
            keyword arguments for :meth:`moscot.problems.CompoundProblem._apply`.

        Returns
        -------
        TODO.
        """
        return self._apply(*args, forward=True, **kwargs)

    @d.get_sections(base="BaseCompoundProblem_pull", sections=["Parameters", "Raises"])
    @d.dedent
    def pull(self, *args: Any, **kwargs: Any) -> ApplyOutput_t[K]:
        """
        Pull mass from `end` to `start`.

        TODO: expose kwargs.

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
            Keyword arguments for :meth:`moscot.problems.CompoundProblem._apply`.

        Returns
        -------
        TODO.
        """
        return self._apply(*args, forward=False, **kwargs)

    @property
    def problems(self) -> Dict[Tuple[K, K], B]:
        """Return dictionary of OT problems which the biological problem consists of."""
        if self._problem_manager is None:
            return {}
        return self._problem_manager.problems

    @require_prepare
    def add_problem(self, key: Tuple[K, K], problem: B, *, overwrite: bool = False) -> "BaseCompoundProblem[K, B]":
        """Add problem."""
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        self._problem_manager.add_problem(key, problem, overwrite=overwrite)
        return self

    @require_prepare
    def remove_problem(self, key: Tuple[K, K]) -> "BaseCompoundProblem[K, B]":
        """Remove problem."""
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        self._problem_manager.remove_problem(key)
        return self

    @property
    def solutions(self) -> Dict[Tuple[K, K], BaseSolverOutput]:
        """Return dictionary of solutions of OT problems which the biological problem consists of."""
        if self._problem_manager is None:
            return {}
        return self._problem_manager.solutions

    @property
    def _policy(self) -> Optional[SubsetPolicy[K]]:
        if self._problem_manager is None:
            return None
        return self._problem_manager._policy

    def __getitem__(self, item: Tuple[K, K]) -> B:
        return self.problems[item]

    def __contains__(self, key: Tuple[K, K]) -> bool:
        return key in self.problems

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self) -> Iterator[Tuple[K, K]]:
        return iter(self.problems)


@d.get_sections(base="CompoundProblem", sections=["Parameters", "Raises"])
@d.dedent
class CompoundProblem(BaseCompoundProblem[K, B], ABC):
    """
    Class handling biological problems composed of exactly one :class:`anndata.AnnData` instance.

    This class is needed to apply the `policy` to one :class:`anndata.AnnData` objects and hence create the
    Optimal Transport subproblems from the biological problem.

    Parameters
    ----------
    %(BaseCompoundProblem.parameters)s

    Raises
    ------
    %(BaseCompoundProblem.raises)s
    """

    @property
    @abstractmethod
    def _base_problem_type(self) -> Type[B]:
        pass

    def _create_problem(self, src_mask: ArrayLike, tgt_mask: ArrayLike, **kwargs: Any) -> B:
        return self._base_problem_type(
            self._mask(src_mask),
            self._mask(tgt_mask),
            **kwargs,
        )

    def _create_policy(
        self,
        policy: Policy,
        key: Optional[str] = None,
        axis: Axis_t = "obs",
        **_: Any,
    ) -> SubsetPolicy[K]:
        if isinstance(policy, str):
            return SubsetPolicy.create(policy, adata=self.adata, key=key, axis=axis)
        return ExplicitPolicy(self.adata, key=key, axis=axis)

    def _mask(self, mask: ArrayLike) -> AnnData:
        return self.adata[mask] if self._policy.axis == "obs" else self.adata[:, mask]  # type: ignore[union-attr]

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
    ) -> Mapping[str, Any]:
        if TYPE_CHECKING:
            assert isinstance(self._policy, SubsetPolicy)

        attr = f"{self._policy.axis}p"
        try:
            data = getattr(self.adata, attr)[key]
        except KeyError:
            raise KeyError(f"TODO: data not in `adata.{attr}[{key!r}]`") from None

        src_mask = self._policy.create_mask(src, allow_empty=False)
        tgt_mask = self._policy.create_mask(tgt, allow_empty=False)

        if return_linear:
            linear_cost_matrix = data[src_mask, :][:, tgt_mask]
            return {
                "xy": TaggedArray(
                    linear_cost_matrix.A if issparse(linear_cost_matrix) else linear_cost_matrix, tag=Tag.COST_MATRIX
                )
            }

        return {
            "x": TaggedArray(data[src_mask, :][:, src_mask], tag=Tag.COST_MATRIX),
            "y": TaggedArray(data[tgt_mask, :][:, tgt_mask], tag=Tag.COST_MATRIX),
        }

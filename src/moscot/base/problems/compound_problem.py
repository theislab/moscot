import abc
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import scipy.sparse as sp

from anndata import AnnData

from moscot._logging import logger
from moscot._types import ArrayLike, Policy_t, ProblemStage_t
from moscot.base.output import BaseSolverOutput
from moscot.base.problems._utils import attributedispatch, require_prepare
from moscot.base.problems.manager import ProblemManager
from moscot.base.problems.problem import BaseProblem, OTProblem
from moscot.utils.subset_policy import (
    DummyPolicy,
    ExplicitPolicy,
    FormatterMixin,
    OrderedPolicy,
    StarPolicy,
    SubsetPolicy,
    create_policy,
)
from moscot.utils.tagged_array import Tag, TaggedArray

__all__ = ["BaseCompoundProblem", "CompoundProblem"]

K = TypeVar("K", bound=Hashable)
B = TypeVar("B", bound=OTProblem)
Callback_t = Callable[
    [Literal["xy", "x", "y"], AnnData, Optional[AnnData]], Mapping[Literal["xy", "x", "y"], TaggedArray]
]
ApplyOutput_t = Union[ArrayLike, Dict[K, ArrayLike]]


class BaseCompoundProblem(BaseProblem, abc.ABC, Generic[K, B]):
    """Base class for all biological problems.

    This class translates a biological problem to multiple :term:`OT` problems.

    Parameters
    ----------
    adata
        Annotated data object.
    kwargs
        Keyword arguments for :class:`~moscot.base.problems.BaseProblem`.
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(**kwargs)
        self._adata = adata
        self._problem_manager: Optional[ProblemManager[K, B]] = None

    @abc.abstractmethod
    def _create_problem(self, src: K, tgt: K, src_mask: ArrayLike, tgt_mask: ArrayLike, **kwargs: Any) -> B:
        """Create an :term:`OT` subproblem.

        Parameters
        ----------
        src
            Source key identifying the subproblem.
        tgt
            Target key identifying the subproblem.
        src_mask
            Source mask used to subset :attr:`adata`.
        tgt_mask
            Target mask used to subset :attr:`adata`.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        The subproblem.
        """

    @abc.abstractmethod
    def _create_policy(
        self,
        policy: Policy_t,
        **kwargs: Any,
    ) -> SubsetPolicy[K]:
        """Create a policy used to split :attr:`adata`.

        Only policies specified by :attr:`_valid_policies` will be passed to this function.

        Parameters
        ----------
        policy
            Name of the policy.
        kwargs
            Keyword arguments for :class:`~moscot.utils.subset_policy.SubsetPolicy`.

        Returns
        -------
        The policy.
        """

    @property
    @abc.abstractmethod
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        """Valid policies for this problem."""

    def _callback_handler(
        self,
        term: Literal["xy", "x", "y"],
        key_1: K,
        key_2: K,
        problem: B,
        *,
        callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        **kwargs: Any,
    ) -> Mapping[Literal["xy", "x", "y"], TaggedArray]:
        def verify_data(data: Mapping[Literal["xy", "x", "y"], TaggedArray]) -> None:
            keys = ("xy", "x", "y")
            for key, val in data.items():
                if key not in keys:
                    raise ValueError(f"Expected key to be one of `{keys}`, found `{key!r}`.")
                if not isinstance(val, TaggedArray):
                    raise TypeError(f"Expected value for `{key}` to be a `TaggedArray`, found `{type(val)}`.")

        if callback is None:
            return {}
        if callback == "local-pca":
            callback = problem._local_pca_callback
        if callback == "spatial-norm":
            callback = problem._spatial_norm_callback

        if not callable(callback):
            raise TypeError("Callback is not a function.")
        data = callback(term, problem.adata_src, problem.adata_tgt, **kwargs)
        verify_data(data)
        return data

    # TODO(michalk8): refactor me
    def _create_problems(
        self,
        xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        x_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        y_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        x_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        y_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> Dict[Tuple[K, K], B]:
        from moscot.base.problems.birth_death import BirthDeathProblem

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

            problem = self._create_problem(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

            xy_data = self._callback_handler(
                term="xy", key_1=src, key_2=tgt, problem=problem, callback=xy_callback, **xy_callback_kwargs
            )

            x_data = self._callback_handler(
                term="x", key_1=src, key_2=tgt, problem=problem, callback=x_callback, **x_callback_kwargs
            )

            y_data = self._callback_handler(
                term="y", key_1=src, key_2=tgt, problem=problem, callback=y_callback, **y_callback_kwargs
            )

            kws = {**kwargs, **xy_data, **x_data, **y_data}  # type: ignore[misc]

            if isinstance(problem, BirthDeathProblem):
                kws["proliferation_key"] = self.proliferation_key  # type: ignore[attr-defined]
                kws["apoptosis_key"] = self.apoptosis_key  # type: ignore[attr-defined]
            problems[src_name, tgt_name] = problem.prepare(**kws)

        return problems

    def prepare(
        self,
        policy: Policy_t,
        key: Optional[str],
        subset: Optional[Sequence[Tuple[K, K]]] = None,
        reference: Optional[Any] = None,
        xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        x_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        y_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        x_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        y_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> "BaseCompoundProblem[K, B]":
        """Prepare the individual :term:`OT` subproblems.

        .. seealso::
            - See :doc:`../notebooks/examples/problems/400_subset_policy` on how to use different policies.

        Parameters
        ----------
        policy
            Rule which defines how to construct the subproblems.
        key
            Key in :attr:`~anndata.AnnData.obs` for the :class:`~moscot.utils.subset_policy.SubsetPolicy`.
        subset
            Subset of :attr:`obs['{key}'] <anndata.AnnData.obs>`
            for the :class:`~moscot.utils.subset_policy.ExplicitPolicy`. Only used when ``policy = 'explicit'``.
        reference
            Reference for the :class:`~moscot.utils.subset_policy.SubsetPolicy`. Only used when ``policy = 'star'``.
        xy_callback
            Callback function used to prepare the data in the :term:`linear term`.
        x_callback
            Callback function used to prepare the data in the source :term:`quadratic term`.
        y_callback
            Callback function used to prepare the data in the target :term:`quadratic term`.
        xy_callback_kwargs
            Keyword arguments for the ``xy_callback``.
        x_callback_kwargs
            Keyword arguments for the ``x_callback``.
        y_callback_kwargs
            Keyword arguments for the ``y_callback``.
        kwargs
            Keyword arguments for the subproblems' :meth:`~moscot.base.problems.OTProblem.prepare` method.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`problems` - the prepared subproblems.
        - :attr:`solutions` - set to an empty :class:`dict`.
        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - kind of the :term:`OT` problem.
        """
        self._ensure_valid_policy(policy)
        policy = self._create_policy(policy=policy, key=key)
        if TYPE_CHECKING:
            assert isinstance(policy, SubsetPolicy)

        if isinstance(policy, ExplicitPolicy):
            policy = policy.create_graph(subset=subset)
        elif isinstance(policy, StarPolicy):
            policy = policy.create_graph(reference=reference)
        else:
            policy = policy.create_graph()

        # TODO(michalk8): manager must be currently instantiated first, since `_create_problems` accesses the policy
        # when refactoring the callback, consider changing this
        self._problem_manager = ProblemManager(self, policy=policy)
        problems = self._create_problems(
            xy_callback=xy_callback,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback_kwargs=xy_callback_kwargs,
            x_callback_kwargs=x_callback_kwargs,
            y_callback_kwargs=y_callback_kwargs,
            **kwargs,
        )
        self._problem_manager.add_problems(problems)

        # we assume that all subproblems are of the same kind
        for p in self.problems.values():
            self._problem_kind = p._problem_kind
            self._stage = "prepared"
            break
        return self

    def solve(
        self,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        **kwargs: Any,
    ) -> "BaseCompoundProblem[K, B]":
        """Solve the individual :term:`OT` subproblems.

        .. seealso:
            - See :doc:`../notebooks/examples/solvers/100_linear_problems_basic`
            for an introduction on how to solve linear problems.
            - See :doc:`../notebooks/examples/solvers/300_quad_problems_basic`
            for an introduction on how to solve quadratic problems.

        Parameters
        ----------
        stage
            Stage by which to filter the :attr:`problems` to be solved.
        kwargs
            Keyword arguments for the subproblems' :meth:`~moscot.base.problems.OTProblem.solve` method.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`solutions` - the :term:`OT` solutions for each subproblem.
        - :attr:`stage` - set to ``'solved'``.
        """
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        problems = self._problem_manager.get_problems(stage=stage)

        logger.info(f"Solving `{len(problems)}` problems")
        for problem in problems.values():
            logger.info(f"Solving problem {problem}.")
            _ = problem.solve(**kwargs)

        self._stage = "solved"
        return self

    @attributedispatch(attr="_policy")
    def _apply(self, *_args: Any, **_kwargs: Any) -> ApplyOutput_t[K]:
        raise NotImplementedError(type(self._policy))

    @_apply.register(DummyPolicy)
    @_apply.register(StarPolicy)
    def _(
        self,
        source: Optional[K] = None,
        target: Optional[K] = None,
        data: Optional[Union[str, ArrayLike]] = None,
        forward: bool = True,
        scale_by_marginals: bool = False,
        return_all: bool = False,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        del target
        if TYPE_CHECKING:
            assert isinstance(self._policy, StarPolicy)

        res = {}
        source = source if isinstance(source, list) else [source]
        for src, tgt in self._policy.plan(
            explicit_steps=kwargs.pop("explicit_steps", None),
            filter=source,  # type: ignore [arg-type]
        ):
            problem = self.problems[src, tgt]
            fun = problem.push if forward else problem.pull
            res[src] = fun(data=data, scale_by_marginals=scale_by_marginals)
        return res if return_all else res[src]

    @_apply.register(ExplicitPolicy)
    @_apply.register(OrderedPolicy)
    def _(
        self,
        source: Optional[K] = None,
        target: Optional[K] = None,
        data: Optional[Union[str, ArrayLike]] = None,
        forward: bool = True,
        scale_by_marginals: bool = False,
        return_all: bool = False,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        explicit_steps = kwargs.pop(
            "explicit_steps", [[source, target]] if isinstance(self._policy, ExplicitPolicy) else None
        )
        if TYPE_CHECKING:
            assert isinstance(self._policy, OrderedPolicy)
        (src, tgt), *rest = self._policy.plan(
            forward=forward,
            start=source,
            end=target,
            explicit_steps=explicit_steps,
        )
        problem = self.problems[src, tgt]
        adata = problem.adata_src if forward else problem.adata_tgt
        current_mass = problem._get_mass(adata, data=data, **kwargs)
        res = {src if forward else tgt: current_mass}
        for _src, _tgt in [(src, tgt)] + rest:
            problem = self.problems[_src, _tgt]
            fun = problem.push if forward else problem.pull
            res[_tgt if forward else _src] = current_mass = fun(current_mass, scale_by_marginals=scale_by_marginals)

        return res if return_all else current_mass

    # TODO(michalk8): better description of `source/target` (also in other places).
    def push(self, *args: Any, **kwargs: Any) -> ApplyOutput_t[K]:
        """Push mass from source to target.

        TODO.
        """
        _ = kwargs.pop("return_data", None)
        _ = kwargs.pop("key_added", None)  # this should be handled by overriding method
        return self._apply(*args, forward=True, **kwargs)

    def pull(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        """Pull mass from target to source.

        TODO
        """
        _ = kwargs.pop("return_data", None)
        _ = kwargs.pop("key_added", None)  # this should be handled by overriding method
        return self._apply(*args, forward=False, **kwargs)

    @property
    def problems(self) -> Dict[Tuple[K, K], B]:
        """:term:`OT` subproblems that define the biological problem."""
        if self._problem_manager is None:
            return {}
        return self._problem_manager.problems

    @require_prepare
    def add_problem(
        self,
        key: Tuple[K, K],
        problem: B,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> "BaseCompoundProblem[K, B]":
        """Add a subproblem.

        .. seealso::
            - See :doc:`../notebooks/examples/problems/300_adding_and_removing_problems` on how to add subproblems.

        Parameters
        ----------
        key
            Key in :attr:`problems` where to add the subproblem.
        problem
            Subproblem to add.
        overwrite
            Whether ot overwrite an existing subproblem in :attr:`problems`.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        Self and updates the following fields:

        - :attr:`problems`
        """
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        self._problem_manager.add_problem(key, problem, overwrite=overwrite, **kwargs)
        return self

    @require_prepare
    def remove_problem(self, key: Tuple[K, K]) -> "BaseCompoundProblem[K, B]":
        """Remove a subproblem.

        .. seealso::
            - See :doc:`../notebooks/examples/problems/300_adding_and_removing_problems` on how to remove subproblems.

        Parameters
        ----------
        key
            Key of the subproblem to remove.

        Returns
        -------
        Self and updates the following fields:

        - :attr:`problems`
        """
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        self._problem_manager.remove_problem(key)
        return self

    @property
    def solutions(self) -> Dict[Tuple[K, K], BaseSolverOutput]:
        """Solutions to the :attr:`problems`."""
        if self._problem_manager is None:
            return {}
        return self._problem_manager.solutions

    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self._adata

    @property
    def _policy(self) -> Optional[SubsetPolicy[K]]:
        if self._problem_manager is None:
            return None
        return self._problem_manager.policy

    def _ensure_valid_policy(self, policy: Policy_t) -> None:
        if self._valid_policies and policy not in self._valid_policies:
            raise ValueError(f"Invalid policy `{policy!r}`. Valid policies are: `{self._valid_policies}`.")

    def __getitem__(self, item: Tuple[K, K]) -> B:
        return self.problems[item]

    def __contains__(self, key: Tuple[K, K]) -> bool:
        return key in self.problems

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self) -> Iterator[Tuple[K, K]]:
        return iter(self.problems)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{list(self.problems.keys())}"

    def __str__(self) -> str:
        return repr(self)


class CompoundProblem(BaseCompoundProblem[K, B], abc.ABC):
    """Base class for all biological problems.

    This class translates a biological problem to multiple :term:`OT` problems.

    Parameters
    ----------
    adata
        Annotated data object.
    kwargs
        Keyword arguments for :class:`~moscot.base.problems.BaseCompoundProblem`.
    """

    @property
    @abc.abstractmethod
    def _base_problem_type(self) -> Type[B]:
        pass

    def _create_problem(self, src: K, tgt: K, src_mask: ArrayLike, tgt_mask: ArrayLike, **kwargs: Any) -> B:
        return self._base_problem_type(
            self.adata, src_obs_mask=src_mask, tgt_obs_mask=tgt_mask, src_key=src, tgt_key=tgt, **kwargs
        )

    def _create_policy(
        self,
        policy: Policy_t,
        key: Optional[str] = None,
        **_: Any,
    ) -> SubsetPolicy[K]:
        if isinstance(policy, str):
            return create_policy(policy, adata=self.adata, key=key)
        return ExplicitPolicy(self.adata, key=key)

    def _callback_handler(
        self,
        term: Literal["xy", "x", "y"],
        key_1: K,
        key_2: K,
        problem: B,
        *,
        callback: Optional[Union[Literal["local-pca", "cost-matrix"], Callback_t]] = None,
        **kwargs: Any,
    ) -> Mapping[Literal["xy", "x", "y"], TaggedArray]:
        if callback == "cost-matrix":
            return self._cost_matrix_callback(term=term, key_1=key_1, key_2=key_2, **kwargs)
        return super()._callback_handler(
            term=term, key_1=key_1, key_2=key_2, problem=problem, callback=callback, **kwargs
        )

    def _cost_matrix_callback(
        self, term: Literal["xy", "x", "y"], *, key: str, key_1: K, key_2: Optional[K] = None, **_: Any
    ) -> Mapping[Literal["xy", "x", "y"], TaggedArray]:
        if TYPE_CHECKING:
            assert isinstance(self._policy, SubsetPolicy)

        try:
            data = self.adata.obsp[key]
        except KeyError:
            raise KeyError(f"Unable to fetch data from `adata.obsp[{key!r}]`.") from None

        mask = self._policy.create_mask(key_1, allow_empty=False)

        if term == "xy":
            if key_2 is None:
                raise ValueError("If `term` is `xy`, `key_2` cannot be `None`.")
            mask_2 = self._policy.create_mask(key_2, allow_empty=False)

            linear_cost_matrix = data[mask, :][:, mask_2]
            if sp.issparse(linear_cost_matrix):
                logger.warning("Linear cost matrix being densified.")
                linear_cost_matrix = linear_cost_matrix.A
            return {"xy": TaggedArray(linear_cost_matrix, tag=Tag.COST_MATRIX)}

        if term in ("x", "y"):
            quad_cost_matrix = data[mask, :][:, mask]
            if sp.issparse(quad_cost_matrix):
                logger.warning("Quadratic cost matrix being densified.")
                quad_cost_matrix = quad_cost_matrix.A
            return {term: TaggedArray(quad_cost_matrix, tag=Tag.COST_MATRIX)}

        raise ValueError(f"Expected `term` to be one of `x`, `y`, or `xy`, found `{term!r}`.")

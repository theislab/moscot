import abc
import os
from types import MappingProxyType
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

import cloudpickle

import scipy.sparse as sp

from anndata import AnnData

from moscot._docs._docs import d
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
# TODO(michalk8): future behavior
# ApplyOutput_t = Union[ArrayLike, Dict[Tuple[K, K], ArrayLike]]


@d.get_sections(base="BaseCompoundProblem", sections=["Parameters", "Raises"])
@d.dedent
class BaseCompoundProblem(BaseProblem, abc.ABC, Generic[K, B]):
    """
    Base class for all biological problems.

    This base class translates a biological problem to potentially multiple Optimal Transport problems.

    Parameters
    ----------
    %(adata)s

    Raises
    ------
    TypeError
        If `base_problem_type` is not a subclass of :class:`~moscot.base.problems.OTProblem`.
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(**kwargs)
        self._adata = adata
        self._problem_manager: Optional[ProblemManager[K, B]] = None

    @abc.abstractmethod
    def _create_problem(self, src: K, tgt: K, src_mask: ArrayLike, tgt_mask: ArrayLike, **kwargs: Any) -> B:
        pass

    @abc.abstractmethod
    def _create_policy(
        self,
        policy: Policy_t,
        **kwargs: Any,
    ) -> SubsetPolicy[K]:
        pass

    @property
    @abc.abstractmethod
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        pass

    # TODO(michalk8): refactor me
    def _callback_handler(
        self,
        term: Literal["x", "y", "xy"],
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
        xy_callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        x_callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        y_callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
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

            kws = {**kwargs, **xy_data, **x_data, **y_data}  # type: ignore[arg-type]

            if isinstance(problem, BirthDeathProblem):
                kws["proliferation_key"] = self.proliferation_key  # type: ignore[attr-defined]
                kws["apoptosis_key"] = self.apoptosis_key  # type: ignore[attr-defined]
            problems[src_name, tgt_name] = problem.prepare(**kws)

        return problems

    @d.get_sections(base="BaseCompoundProblem_prepare", sections=["Parameters", "Raises"])
    @d.dedent
    def prepare(
        self,
        key: str,
        policy: Policy_t = "sequential",
        subset: Optional[Sequence[Tuple[K, K]]] = None,
        reference: Optional[Any] = None,
        xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        x_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        y_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        xy_callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        x_callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        y_callback_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "BaseCompoundProblem[K, B]":
        """
        Prepare the biological problem.

        Parameters
        ----------
        %(key)s
        %(policy)s
        %(subset)s
        %(reference)s
        %(xy_callback)s
        %(x_callback)s
        %(y_callback)s
        %(xy_callback_kwargs)s
        %(x_callback_kwargs)s
        %(y_callback_kwargs)s
        %(a)s
        %(b)s

        Returns
        -------
        The prepared problem.
        """
        self._ensure_valid_policy(policy)
        policy = self._create_policy(policy=policy, key=key)
        if TYPE_CHECKING:
            assert isinstance(policy, SubsetPolicy)

        if isinstance(policy, ExplicitPolicy):
            policy = policy(subset=subset)
        elif isinstance(policy, StarPolicy):
            policy = policy(reference=reference)
        else:
            policy = policy()

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

        for p in self.problems.values():
            self._problem_kind = p._problem_kind
            break
        return self

    def solve(
        self,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        **kwargs: Any,
    ) -> "BaseCompoundProblem[K,B]":
        """
        Solve the biological problem.

        Parameters
        ----------
        stage
            Some stage TODO.
        kwargs
            Keyword arguments for :meth:`~moscot.base.problems.OTProblem.solve`.

        Returns
        -------
        The solver problem.
        """
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        problems = self._problem_manager.get_problems(stage=stage)

        logger.info(f"Solving `{len(problems)}` problems")
        for _, problem in problems.items():
            logger.info(f"Solving problem {problem}.")
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
        source: Optional[K] = None,
        return_all: bool = True,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        if TYPE_CHECKING:
            assert isinstance(self._policy, StarPolicy)
        res = {}
        # TODO(michalk8): should use manager.plan (once implemented), as some problems may not be solved
        # TODO: better check
        source = source if isinstance(source, list) else [source]
        _ = kwargs.pop("target", None)  # make compatible with Explicit/Ordered policy
        for src, tgt in self._policy.plan(
            explicit_steps=kwargs.pop("explicit_steps", None),
            filter=source,  # type: ignore [arg-type]
        ):
            problem = self.problems[src, tgt]
            fun = problem.push if forward else problem.pull
            res[src] = fun(data=data, scale_by_marginals=scale_by_marginals, **kwargs)
        return res if return_all else res[src]

    @_apply.register(ExplicitPolicy)
    @_apply.register(OrderedPolicy)
    def _(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        forward: bool = True,
        scale_by_marginals: bool = False,
        source: Optional[K] = None,
        target: Optional[K] = None,
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
        # TODO(michlak8): future behavior
        # res = {(None, src) if forward else (tgt, None): current_mass}
        res = {src if forward else tgt: current_mass}
        for _src, _tgt in [(src, tgt)] + rest:
            problem = self.problems[_src, _tgt]
            fun = problem.push if forward else problem.pull
            res[_tgt if forward else _src] = current_mass = fun(
                current_mass, scale_by_marginals=scale_by_marginals, **kwargs
            )

        return res if return_all else current_mass

    @d.get_sections(base="BaseCompoundProblem_push", sections=["Parameters", "Raises"])
    @d.dedent  # TODO(@MUCDK) document private _apply
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
            If `True` and transport maps are applied consecutively only the final mass is returned.
            Otherwise, all intermediate step results are returned, too.

        %(scale_by_marginals)s

        kwargs
            keyword arguments for policy-specific `_apply` method of :class:`moscot.base.problems.CompoundProblem`.

        Returns
        -------
        TODO.
        """
        _ = kwargs.pop("return_data", None)
        _ = kwargs.pop("key_added", None)  # this should be handled by overriding method
        return self._apply(*args, forward=True, **kwargs)

    @d.get_sections(base="BaseCompoundProblem_pull", sections=["Parameters", "Raises"])
    @d.dedent  # TODO(@MUCDK) document private functions
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
            keyword arguments for policy-specific `_apply` method of :class:`moscot.base.problems.CompoundProblem`.

        Returns
        -------
        TODO.
        """
        _ = kwargs.pop("return_data", None)
        _ = kwargs.pop("key_added", None)  # this should be handled by overriding method
        return self._apply(*args, forward=False, **kwargs)

    @property
    def problems(self) -> Dict[Tuple[K, K], B]:
        """Return dictionary of OT problems which the biological problem consists of."""
        if self._problem_manager is None:
            return {}
        return self._problem_manager.problems

    @d.dedent
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

        This function adds and prepares a problem, e.g. if it is not included by the initial
        :class:`~moscot.utils.subset_policy.SubsetPolicy`.

        Parameters
        ----------
        %(key)s

        problem
            Instance of a :class:`~moscot.base.problems.OTProblem`.
        overwrite
            If `True` the problem will be reinitialized and prepared even if a problem with `key` exists.

        Returns
        -------
        The updated compound problem.
        """
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        self._problem_manager.add_problem(key, problem, overwrite=overwrite, **kwargs)
        return self

    @d.dedent
    @require_prepare
    def remove_problem(self, key: Tuple[K, K]) -> "BaseCompoundProblem[K, B]":
        """Remove a subproblem.

        Parameters
        ----------
        %(key)s

        Returns
        -------
        The updated compound problem.
        """
        if TYPE_CHECKING:
            assert isinstance(self._problem_manager, ProblemManager)
        self._problem_manager.remove_problem(key)
        return self

    # TODO(MUCKD): should be on the OT problem level as well
    def save(
        self,
        dir_path: str,
        file_prefix: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save the model.

        As of now this method pickled the problem class instance. Modifications depend on the I/O of the backend.

        Parameters
        ----------
        dir_path
            Path to a directory, defaults to current directory
        file_prefix
            Prefix to prepend to the file name.
        overwrite
            Whether to overwrite existing data or not.

        Returns
        -------
        Nothing, just saves the problem.
        """
        file_name = (
            f"{file_prefix}_{self.__class__.__name__}.pkl"
            if file_prefix is not None
            else f"{self.__class__.__name__}.pkl"
        )
        file_dir = os.path.join(dir_path, file_name) if dir_path is not None else file_name

        if not overwrite and os.path.exists(file_dir):
            raise RuntimeError(f"Unable to save to an existing file `{file_dir}` use `overwrite=True` to overwrite it.")
        with open(file_dir, "wb") as f:
            cloudpickle.dump(self, f)

        logger.info(f"Successfully saved the problem as `{file_dir}`")

    # TODO(MUCKD): should be on the OT problem level as well
    @classmethod
    def load(
        cls,
        filename: str,
    ) -> "BaseCompoundProblem[K, B]":
        """
        Instantiate a moscot problem from a saved output.

        Parameters
        ----------
        filename
            filename of the model to load

        Returns
        -------
        Loaded instance of the model.

        Examples
        --------
        #TODO(michalk8): make nicer
        >>> problem = ProblemClass.load(filename) # use the name of the model class used to save
        >>> problem.push....
        """
        with open(filename, "rb") as f:
            problem = cloudpickle.load(f)
        if type(problem) is not cls:
            raise TypeError(f"Expected the problem to be type of `{cls}`, found `{type(problem)}`.")
        return problem

    @property
    def solutions(self) -> Dict[Tuple[K, K], BaseSolverOutput]:
        """Return dictionary of solutions of OT problems which the biological problem consists of."""
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
        return self._problem_manager._policy

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


@d.get_sections(base="CompoundProblem", sections=["Parameters", "Raises"])
@d.dedent
class CompoundProblem(BaseCompoundProblem[K, B], abc.ABC):
    """Class handling biological problems composed of exactly one :class:`~anndata.AnnData` instance.

    This class is needed to apply the `policy` to one :class:`~anndata.AnnData` objects and hence create the
    Optimal Transport subproblems from the biological problem.

    Parameters
    ----------
    %(BaseCompoundProblem.parameters)s
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
        # TODO(michalk8): better name?

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

from typing import (
    Any,
    Generic,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from anndata import AnnData

from moscot import backends
from moscot._logging import logger
from moscot._types import ArrayLike, Device_t
from moscot.base.output import BaseNeuralOutput
from moscot.base.problems._utils import wrap_prepare, wrap_solve
from moscot.base.problems.problem import BaseProblem
from moscot.base.solver import OTSolver
from moscot.neural.data import DistributionCollection, NeuralDistribution
from moscot.utils.subset_policy import (  # type:ignore[attr-defined]
    ExplicitPolicy,
    Policy_t,
    StarPolicy,
    SubsetPolicy,
    create_policy,
)

K = TypeVar("K", bound=Hashable)

__all__ = ["NeuralOTProblem"]


class NeuralOTProblem(BaseProblem, Generic[K]):  # TODO(@MUCDK) check generic types, save and load
    """
    Base class for all conditional (nerual) optimal transport problems.

    Parameters
    ----------
    adata
        Source annotated data object.
    kwargs
        Keyword arguments for :class:`moscot.base.problems.problem.BaseProblem`
    """

    def __init__(
        self,
        adata: AnnData,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._adata = adata

        self._distributions: Optional[DistributionCollection[K]] = None
        self._policy: Optional[SubsetPolicy[Any]] = None

        self._solver: Optional[OTSolver[BaseNeuralOutput]] = None
        self._solution: Optional[BaseNeuralOutput] = None

        self._a: Optional[str] = None
        self._b: Optional[str] = None

    @wrap_prepare
    def prepare(
        self,
        policy_key: str,
        policy: Policy_t,
        lin: Mapping[str, Any],
        src_quad: Optional[Mapping[str, Any]] = None,
        tgt_quad: Optional[Mapping[str, Any]] = None,
        condition: Optional[Mapping[str, Any]] = None,
        subset: Optional[Sequence[Tuple[K, K]]] = None,
        seed: int = 0,
        reference: K = None,
    ) -> "NeuralOTProblem[K]":
        """Prepare conditional optimal transport problem.

        Parameters
        ----------
        xy
            Geometry defining the linear term. If passed as a :class:`dict`,
            :meth:`~moscot.utils.tagged_array.TaggedArray.from_adata` will be called.
        policy
            Policy defining which pairs of distributions to sample from during training.
        policy_key
            %(key)s
        kwargs
            Keyword arguments when creating the source/target marginals.


        Returns
        -------
        Self and modifies the following attributes:
        TODO.
        """
        self._seed = seed
        self._problem_kind = "linear"
        self._distributions = DistributionCollection()
        self._solution = None
        self._policy_key = policy_key
        try:
            self._distribution_id = pd.Series(self.adata.obs[policy_key])
        except KeyError:
            raise KeyError(f"Unable to find data in `adata.obs[{policy_key!r}]`.") from None

        self._policy = create_policy(policy, adata=self.adata, key=policy_key)
        if isinstance(self._policy, ExplicitPolicy):
            self._policy = self._policy.create_graph(subset=subset)
        elif isinstance(self._policy, StarPolicy):
            self._policy = self._policy.create_graph(reference=reference)
        else:
            _ = self.policy.create_graph()  # type: ignore[union-attr]

        if src_quad is None and tgt_quad is not None:
            raise ValueError("If `tgt_quad` is provided, `src_quad` must also be provided.")
        if src_quad is not None and tgt_quad is None:
            raise ValueError("If `src_quad` is provided, `tgt_quad` must also be provided.")
        if src_quad is not None:
            # which edges will be always source
            source_nodes = {el[0] for el in self.policy.plan()}  # type: ignore[union-attr]
            target_nodes = {el[1] for el in self.policy.plan()}  # type: ignore[union-attr]
            # if there aren't nodes that are always source or target, we will warn the user
            # that we will choose source quad attributes
            tgt_quad_nodes = target_nodes - source_nodes
            if not source_nodes.isdisjoint(target_nodes):
                logger.warning(
                    "Some nodes are both source and target in the policy plan, "
                    "we will choose source quad attributes for such nodes."
                )
        for el in self.policy.categories:  # type: ignore[union-attr]
            adata_masked = self.adata[self._create_mask(el)]
            # TODO: Marginals
            quad = None
            if src_quad is not None:
                quad = tgt_quad if el in tgt_quad_nodes else src_quad
            self.distributions[el] = NeuralOTProblem._create_neural_distribution(  # type: ignore[index]
                adata_masked,
                lin=lin,
                quad=quad,
                condition=condition,
            )
        return self

    @wrap_solve
    def solve(
        self,
        backend: Literal["neural_ott"] = "neural_ott",
        solver_name: Literal["GENOTSolver"] = "GENOTSolver",
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> "NeuralOTProblem[K]":
        """Solve optimal transport problem.

        Parameters
        ----------
        backend
            Which backend to use, see :func:`moscot.backends.utils.get_available_backends`.
        device
            Device where to transfer the solution, see :meth:`moscot.base.output.BaseNeuralOutput.to`.
        kwargs
            Keyword arguments for :meth:`moscot.base.solver.BaseSolver.__call__`.


        Returns
        -------
        Self and modifies the following attributes:
        - :attr:`solver`: optimal transport solver.
        - :attr:`solution`: optimal transport solution.
        """
        assert self.distributions is not None
        distributions: DistributionCollection[K] = self.distributions
        assert next(iter(self.distributions.keys())) is not None
        tmp_key: K = next(iter(self.distributions.keys()))
        input_dim = distributions[tmp_key].shared_space.shape[1]
        cond_dim = 0
        if distributions[tmp_key].condition is not None:
            cond_dim = distributions[tmp_key].condition.shape[1]  # type: ignore[union-attr]

        solver_class = backends.get_solver(
            self.problem_kind, solver_name=solver_name, backend=backend, return_class=True
        )
        init_kwargs, call_kwargs = solver_class._partition_kwargs(**kwargs)
        self._solver = solver_class(input_dim=input_dim, cond_dim=cond_dim, **init_kwargs)
        self._solution = self._solver(  # type: ignore[misc]
            device=device,
            distributions=self.distributions,
            policy=self.policy,
            **call_kwargs,
        )

        return self

    # TODO: Marginals
    def _create_marginals(
        self, adata: AnnData, *, source: bool, data: Optional[str] = None, **kwargs: Any
    ) -> ArrayLike:
        if data is True:
            marginals = self.estimate_marginals(adata, source=source, **kwargs)
        elif data in (False, None):
            marginals = np.ones((adata.n_obs,), dtype=float) / adata.n_obs
        elif isinstance(data, str):
            try:
                marginals = np.asarray(adata.obs[data], dtype=float)
            except KeyError:
                raise KeyError(f"Unable to find data in `adata.obs[{data!r}]`.") from None
        return marginals

    def _create_mask(self, value: Union[K, Sequence[K]], *, allow_empty: bool = False) -> ArrayLike:
        """Create a mask used to subset the data.

        TODO(@MUCDK): this is copied from SubsetPolicy, consider making this a function.

        Parameters
        ----------
        value
            Values in the data which determine the mask.
        allow_empty
            Whether to allow empty mask.

        Returns
        -------
        Boolean mask of the same shape as the data.
        """
        if isinstance(value, str) or not isinstance(value, Iterable):
            mask = self._distribution_id == value
        else:
            mask = self._distribution_id.isin(value)
        if not allow_empty and not np.sum(mask):
            raise ValueError("Unable to construct an empty mask, use `allow_empty=True` to override.")
        return np.asarray(mask)

    @property
    def distributions(self) -> Optional[DistributionCollection[K]]:
        """Collection of distributions."""
        return self._distributions

    @property
    def adata(self) -> AnnData:
        """Source annotated data object."""
        return self._adata

    @property
    def solution(self) -> Optional[BaseNeuralOutput]:
        """Solution of the optimal transport problem."""
        return self._solution

    @property
    def solver(self) -> Optional[OTSolver[BaseNeuralOutput]]:
        """Solver of the optimal transport problem."""
        return self._solver

    @property
    def policy(self) -> Optional[SubsetPolicy[Any]]:
        """Policy used to subset the data."""
        return self._policy

    @staticmethod
    def _extract_data(
        adata: AnnData,
        *,
        attr: Literal["X", "obs", "obsp", "obsm", "var", "varm", "layers", "uns"],
        key: Optional[str] = None,
    ) -> jax.Array:
        modifier = f"adata.{attr}" if key is None else f"adata.{attr}[{key!r}]"
        data = getattr(adata, attr)

        try:
            if key is not None:
                data = data[key]
        except KeyError:
            raise KeyError(f"Unable to fetch data from `{modifier}`.") from None
        except IndexError:
            raise IndexError(f"Unable to fetch data from `{modifier}`.") from None

        if attr == "obs":
            data = np.asarray(data)[:, None]
        if sp.issparse(data):
            logger.warning(f"Densifying data in `{modifier}`")
            data = data.toarray()
        if data.ndim != 2:
            raise ValueError(f"Expected `{modifier}` to have `2` dimensions, found `{data.ndim}`.")

        return jnp.array(data)

    @staticmethod
    def _create_neural_distribution(
        adata: AnnData,
        lin: Optional[Mapping[str, Any]] = None,
        quad: Optional[Mapping[str, Any]] = None,
        condition: Optional[Mapping[str, Any]] = None,
    ) -> NeuralDistribution:
        fields = [
            ("shared_space", lin),
            ("incomparable_space", quad),
            ("condition", condition),
        ]
        return NeuralDistribution(
            **{
                field_name: NeuralOTProblem._extract_data(adata, **field)
                for field_name, field in fields
                if field is not None
            }
        )

    @staticmethod
    def _handle_attr(elem: Union[str, Mapping[str, Any]]) -> dict[str, Any]:
        if isinstance(elem, str):
            return {
                "attr": "obsm",
                "key": elem,
            }
        if isinstance(elem, Mapping):
            attr = dict(elem)
            if "attr" not in attr:
                raise KeyError("`attr` must be provided when `attr` is a mapping.")
            if elem["attr"] == "X":
                return {
                    "attr": "X",
                }
            if elem["attr"] in ("obsm", "obsp", "obs", "uns"):
                if "key" not in elem:
                    raise KeyError("`key` must be provided when `attr` is `obsm`, `obsp`, `obs`, or `uns`.")
                return {
                    "attr": elem["attr"],
                    "key": elem["key"],
                }

        raise TypeError(f"Unrecognized `attr` format: {elem}.")

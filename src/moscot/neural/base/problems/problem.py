from typing import (
    Any,
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

import numpy as np
import pandas as pd

from anndata import AnnData

from moscot import backends
from moscot._types import ArrayLike, Device_t
from moscot.base.output import BaseNeuralOutput
from moscot.base.problems._utils import wrap_prepare, wrap_solve
from moscot.base.problems.problem import BaseProblem
from moscot.base.solver import OTSolver
from moscot.neural.data import DistributionCollection, DistributionContainer
from moscot.utils.subset_policy import (  # type:ignore[attr-defined]
    ExplicitPolicy,
    Policy_t,
    StarPolicy,
    SubsetPolicy,
    create_policy,
)

K = TypeVar("K", bound=Hashable)

__all__ = ["NeuralOTProblem"]


class NeuralOTProblem(BaseProblem):  # TODO(@MUCDK) check generic types, save and load
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

        self._distributions: Optional[DistributionCollection[K]] = None  # type: ignore[valid-type]
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
        xy: Mapping[str, Any],
        xx: Mapping[str, Any],
        conditions: Mapping[str, Any],
        subset: Optional[Sequence[Tuple[K, K]]] = None,
        seed: int = 0,
        reference: K = None,
    ) -> "NeuralOTProblem":
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

        for el in self.policy.categories:  # type: ignore[union-attr]
            adata_masked = self.adata[self._create_mask(el)]
            # TODO: Marginals
            self.distributions[el] = DistributionContainer.from_adata(  # type: ignore[index]
                adata_masked, **xy, **xx, **conditions
            )
        return self

    @wrap_solve
    def solve(
        self,
        backend: Literal["neural_ott"] = "neural_ott",
        solver_name: Literal["GENOTSolver"] = "GENOTSolver",
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> "NeuralOTProblem":
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
        tmp = next(iter(self.distributions))  # type: ignore[arg-type]
        input_dim = self.distributions[tmp].xy.shape[1]  # type: ignore[union-attr, index]
        cond_dim = self.distributions[tmp].conditions.shape[1]  # type: ignore[union-attr, index]

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

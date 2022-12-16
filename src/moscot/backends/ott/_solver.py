from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Union, Literal, Mapping, Optional
import math

from scipy.sparse import issparse

from ott.geometry.costs import Bures, Cosine, CostFn, Euclidean, SqEuclidean, UnbalancedBures
from ott.geometry.geometry import Geometry
from ott.solvers.was_solver import WassersteinSolver
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.geometry.epsilon_scheduler import Epsilon
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from ott.problems.linear.linear_problem import LinearProblem
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
import jax
import jax.numpy as jnp

from moscot._types import ArrayLike, QuadInitializer_t, SinkhornInitializer_t
from moscot._constants._enum import ModeEnum
from moscot.backends.ott._output import OTTOutput, NeuralOutput
from moscot.problems.base._utils import _filter_kwargs
from moscot.solvers._base_solver import OTSolver, ProblemKind
from moscot.solvers._tagged_array import TaggedArray
from moscot.backends.ott._jax_data import JaxSampler
from moscot.backends.ott._neuraldual import NeuralDualSolver

__all__ = ["OTTCost", "SinkhornSolver", "GWSolver", "FGWSolver", "NeuralSolver", "CondNeuralSolver"]

Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]
Epsilon_t = Union[float, Epsilon]


class OTTCost(ModeEnum):
    EUCL = "euclidean"
    SQEUCL = "sq_euclidean"
    COSINE = "cosine"
    BURES = "bures"
    BUREL_UNBAL = "unbalanced_bures"

    def __call__(self, **kwargs: Any) -> CostFn:
        if self.value == OTTCost.EUCL:
            return Euclidean()
        if self.value == OTTCost.SQEUCL:
            return SqEuclidean()
        if self.value == OTTCost.COSINE:
            return Cosine()
        if self.value == OTTCost.BURES:
            return Bures(**kwargs)
        if self.value == OTTCost.BUREL_UNBAL:
            return UnbalancedBures(**kwargs)
        raise NotImplementedError(self.value)


class OTTJaxSolver(OTSolver[OTTOutput]):
    """Base class for :mod:`ott` solvers :cite:`cuturi2022optimal`.

    Parameters
    ----------
    jit
        Whether to jit the :attr:`solver`.
    """

    def __init__(self, jit: bool = True):
        super().__init__()
        self._solver: Optional[Union[Sinkhorn, LRSinkhorn, GromovWasserstein]] = None
        self._problem: Optional[Union[LinearProblem, QuadraticProblem]] = None
        self._jit = jit

    def _create_geometry(
        self,
        x: TaggedArray,
        **kwargs: Any,
    ) -> Geometry:
        if x.is_point_cloud:
            kwargs = _filter_kwargs(PointCloud, Geometry, **kwargs)
            cost_fn = self._create_cost(x.cost)
            x, y = self._assert2d(x.data_src), self._assert2d(x.data_tgt)
            n, m = x.shape[1], (None if y is None else y.shape[1])
            if m is not None and n != m:
                raise ValueError(f"Expected `x/y` to have the same number of dimensions, found `{n}/{m}`.")
            return PointCloud(x, y=y, cost_fn=cost_fn, **kwargs)  # TODO: add ScaleCost

        kwargs = _filter_kwargs(Geometry, **kwargs)
        arr = self._assert2d(x.data_src, allow_reshape=False)
        if x.is_cost_matrix:
            return Geometry(cost_matrix=arr, **kwargs)
        if x.is_kernel:
            return Geometry(kernel_matrix=arr, **kwargs)
        raise NotImplementedError(f"Creating geometry from `tag={x.tag!r}` is not yet implemented.")

    def _solve(  # type: ignore[override]
        self,
        prob: Union[LinearProblem, QuadraticProblem],
        **kwargs: Any,
    ) -> OTTOutput:
        solver = jax.jit(self.solver) if self._jit else self._solver
        out = solver(prob, **kwargs)  # type: ignore[misc]
        return OTTOutput(out)

    @staticmethod
    def _assert2d(arr: Optional[ArrayLike], *, allow_reshape: bool = True) -> Optional[jnp.ndarray]:
        if arr is None:
            return None
        arr: jnp.ndarray = jnp.asarray(arr.A if issparse(arr) else arr)  # type: ignore[attr-defined, no-redef]
        if allow_reshape and arr.ndim == 1:
            return jnp.reshape(arr, (-1, 1))
        if arr.ndim != 2:
            raise ValueError(f"Expected array to have 2 dimensions, found `{arr.ndim}`.")
        return arr

    @staticmethod
    def _create_cost(cost: Optional[Union[str, CostFn]], **kwargs: Any) -> CostFn:
        if isinstance(cost, CostFn):
            return cost
        if cost is None:
            cost = "sq_euclidean"
        return OTTCost(cost)(**kwargs)

    @property
    def solver(self) -> Union[Sinkhorn, LRSinkhorn, GromovWasserstein]:
        """Underlying :mod:`ott` solver."""
        return self._solver

    @property
    def rank(self) -> int:
        """Rank of the :attr:`solver`."""
        return getattr(self.solver, "rank", -1)

    @property
    def is_low_rank(self) -> bool:
        """Whether the :attr:`solver` is low-rank."""
        return self.rank > -1


class SinkhornSolver(OTTJaxSolver):
    """Linear optimal transport problem solver.

    The (Kantorovich relaxed) optimal transport problem is defined by two distributions in the same space.
    The aim is to obtain a probabilistic map from the source distribution to the target distribution such that
    the (weighted) sum of the distances between coupled data point in the source and the target distribution is
    minimized.

    Parameters
    ----------
    rank
        Rank of the linear solver. If `-1`, use :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` :cite:`cuturi:2013`,
        otherwise, use :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn` :cite:`scetbon:21a`.
    initializer
        Initializer for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    initializer_kwargs
        Keyword arguments for the initializer.
    kwargs
        Keyword arguments for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    """

    def __init__(
        self,
        rank: int = -1,
        initializer: SinkhornInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__()
        if rank > -1:
            kwargs = _filter_kwargs(Sinkhorn, LRSinkhorn, **kwargs)
            initializer = initializer if initializer is not None else "rank2"  # set rank2 as default LR initializer
            self._solver = LRSinkhorn(rank=rank, initializer=initializer, kwargs_init=initializer_kwargs, **kwargs)
        else:
            kwargs = _filter_kwargs(Sinkhorn, **kwargs)
            initializer = initializer if initializer is not None else "default"  # `None` not handled by backend
            self._solver = Sinkhorn(initializer=initializer, kwargs_init=initializer_kwargs, **kwargs)

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        epsilon: Optional[Epsilon_t] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        cost_matrix_rank: Optional[int] = None,
        **kwargs: Any,
    ) -> LinearProblem:
        del x, y
        if xy is None:
            raise ValueError(f"Unable to create geometry from `xy={xy}`.")

        geom = self._create_geometry(xy, epsilon=epsilon, batch_size=batch_size, scale_cost=scale_cost, **kwargs)
        if self.is_low_rank:

            geom = geom.to_LRCGeometry(
                rank=self.rank if cost_matrix_rank is None else cost_matrix_rank
            )  # batch_size cannot be passed in this function
        kwargs = _filter_kwargs(LinearProblem, **kwargs)
        self._problem = LinearProblem(geom, **kwargs)

        return self._problem

    @property
    def xy(self) -> Optional[Geometry]:
        """Geometry defining the linear term."""
        return None if self._problem is None else self._problem.geom

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.LINEAR


class GWSolver(OTTJaxSolver):
    """Solver solving quadratic optimal transport problem.

    The Gromov-Wasserstein (GW) problem involves two distribution in possibly two different spaces.
    Points in the source distribution are matched to points in the target distribution by comparing the relative
    location of the points within each distribution.

    Parameters
    ----------
    rank
        Rank of the quadratic solver. If `-1` use the full-rank GW :cite:`memoli:2011`,
        otherwise, use the low-rank approach :cite:`scetbon:21b`.
    initializer
        Initializer for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein`.
    gamma
        Only in low-rank setting: the (inverse of the) gradient step size used by the mirror descent algorithm
        (:cite:`scetbon:22b`).
    gamma_rescale
        Only in low-rank setting: whether to rescale `gamma` every iteration as described in :cite:`scetbon:22b`.
    initializer_kwargs
        Keyword arguments for the initializer.
    linear_solver_kwargs
        Keyword arguments for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    kwargs
        Keyword arguments for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein` .
    """

    def __init__(
        self,
        rank: int = -1,
        initializer: QuadInitializer_t = None,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        linear_solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__()
        if rank > -1:
            initializer = initializer if initializer is not None else "rank2"
            linear_ot_solver = LRSinkhorn(
                rank=rank, gamma=gamma, gamma_rescale=gamma_rescale, **linear_solver_kwargs
            )  # initialization handled by quad_initializer
        else:
            initializer = None
            linear_ot_solver = Sinkhorn(**linear_solver_kwargs)
        kwargs = _filter_kwargs(GromovWasserstein, WassersteinSolver, **kwargs)
        self._solver = GromovWasserstein(
            rank=rank,
            linear_ot_solver=linear_ot_solver,
            quad_initializer=initializer,
            kwargs_init=initializer_kwargs,
            **kwargs,
        )

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        **kwargs: Any,
    ) -> QuadraticProblem:
        del xy
        if x is None or y is None:
            raise ValueError(f"Unable to create geometry from `x={x}`, `y={y}`.")
        geom_x = self._create_geometry(x, **kwargs)
        geom_y = self._create_geometry(y, **kwargs)

        kwargs = _filter_kwargs(QuadraticProblem, **kwargs)
        self._problem = QuadraticProblem(geom_x, geom_y, geom_xy=None, **kwargs)
        return self._problem

    @property
    def x(self) -> Optional[Geometry]:
        """First geometry defining the quadratic term."""
        return None if self._problem is None else self._problem.geom_xx

    @property
    def y(self) -> Geometry:
        """Second geometry defining the quadratic term."""
        return None if self._problem is None else self._problem.geom_yy

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD


class FGWSolver(GWSolver):
    """
    Class which solves quadratic OT problem with a linear term included.

    The Fused Gromov-Wasserstein (FGW) problem involves two distributions living in two subspaces,
    corresponding to the linear term and the quadratic term, respectively.

    The subspace corresponding to the linear term is shared between the two distributions.
    The subspace corresponding to the quadratic term is defined in possibly two different spaces.
    The matching obtained from the FGW is a compromise between the ones induced by the linear OT problem and
    the quadratic OT problem :cite:`vayer:2018`.

    This solver wraps :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein`
    with a non-trivial ``fused_penalty``.

    Parameters
    ----------
    rank
        Rank of the quadratic solver. If `-1` use the full-rank GW :cite:`memoli:2011`,
        otherwise, use the low-rank approach :cite:`scetbon:21b`.
    initializer
        `quad_initializer` of :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein`.
    initializer_kwargs
        Keyword arguments for the initializer.
    linear_solver_kwargs
        Keyword arguments for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    kwargs
        Keyword arguments for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein` .
    """

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if xy is None:
            raise ValueError(f"Unable to create geometry from `xy={xy}`.")

        prob = super()._prepare(x=x, y=y, **kwargs)
        geom_xy = self._create_geometry(xy, **kwargs)
        self._validate_geoms(prob.geom_xx, prob.geom_yy, geom_xy)

        kwargs = _filter_kwargs(QuadraticProblem, **kwargs)
        self._problem = QuadraticProblem(
            geom_xx=prob.geom_xx,
            geom_yy=prob.geom_yy,
            geom_xy=geom_xy,
            fused_penalty=self._alpha_to_fused_penalty(alpha),
            **kwargs,
        )
        return self._problem

    @property
    def xy(self) -> Optional[Geometry]:
        """Geometry defining the linear term."""
        return None if self._problem is None else self._problem.geom_xy

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD_FUSED

    @staticmethod
    def _validate_geoms(geom_x: Geometry, geom_y: Geometry, geom_xy: Geometry) -> None:
        n, m = geom_xy.shape
        n_, m_ = geom_x.shape[0], geom_y.shape[0]
        if n != n_:
            raise ValueError(f"Expected the first geometry to have `{n}` points, found `{n_}`.")
        if m != m_:
            raise ValueError(f"Expected the second geometry to have `{m}` points, found `{m_}`.")

    @staticmethod
    def _alpha_to_fused_penalty(alpha: float) -> float:
        if not (0 < alpha <= 1):
            raise ValueError(f"Expected `alpha` to be in interval `(0, 1]`, found `{alpha}`.")
        return (1 - alpha) / alpha


class NeuralSolver(OTSolver[OTTOutput]):
    """Solver class solving Neural Optimal Transport problems."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._train_sampler: Optional[JaxSampler] = None
        self._valid_sampler: Optional[JaxSampler] = None
        kwargs = _filter_kwargs(NeuralDualSolver, **kwargs)
        self._solver = NeuralDualSolver(**kwargs)

    def _prepare(  # type: ignore[override]
        self,
        xy: TaggedArray,
        a: Optional[ArrayLike] = None,
        b: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Tuple[JaxSampler, JaxSampler]:
        if xy.data_tgt is None:
            raise ValueError(f"Unable to obtain target data from `xy={xy}`.")
        x, y = self._assert2d(xy.data_src), self._assert2d(xy.data_tgt)
        n, m = x.shape[1], y.shape[1]
        if n != m:
            raise ValueError(f"Expected `x/y` to have the same number of dimensions, found `{n}/{m}`.")
        train_size = kwargs.pop("train_size", 1.0)
        if train_size > 1.0 or train_size <= 0.0:
            raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")
        if train_size != 1.0:
            seed = kwargs.pop("seed", 0)
            train_x, train_y, valid_x, valid_y, train_a, train_b, valid_a, valid_b = self._split_data(
                x, y, train_size=train_size, seed=seed, a=a, b=b
            )
        else:
            train_x, train_y, train_a, train_b = x, y, a, b
            valid_x, valid_y, valid_a, valid_b = x, y, a, b

        kwargs = _filter_kwargs(JaxSampler, **kwargs)
        self._train_sampler = JaxSampler(
            [train_x, train_y], policies=[(0, 1)], a=[train_a, []], b=[[], train_b], **kwargs
        )
        self._valid_sampler = JaxSampler(
            [valid_x, valid_y], policies=[(0, 1)], a=[valid_a, []], b=[[], valid_b], **kwargs
        )
        return (self._train_sampler, self._valid_sampler)

    def _solve(self, data_samplers: Tuple[JaxSampler, JaxSampler]) -> NeuralOutput:  # type: ignore[override]
        model, logs = self.solver(data_samplers[0], data_samplers[1])
        return NeuralOutput(model, logs)

    @staticmethod
    def _assert2d(arr: ArrayLike, *, allow_reshape: bool = True) -> jnp.ndarray:
        arr: jnp.ndarray = jnp.asarray(arr.A if issparse(arr) else arr)  # type: ignore[no-redef, attr-defined]
        if allow_reshape and arr.ndim == 1:
            return jnp.reshape(arr, (-1, 1))
        if arr.ndim != 2:
            raise ValueError(f"Expected array to have 2 dimensions, found `{arr.ndim}`.")
        return arr

    def _split_data(
        self,
        x: ArrayLike,
        y: ArrayLike,
        train_size: float,
        seed: int,
        a: Optional[ArrayLike] = None,
        b: Optional[ArrayLike] = None,
    ) -> Tuple[
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        Optional[ArrayLike],
        Optional[ArrayLike],
        Optional[ArrayLike],
        Optional[ArrayLike],
    ]:
        n_samples_x = x.shape[0]
        n_samples_y = y.shape[0]
        n_train_x = math.ceil(train_size * n_samples_x)
        n_train_y = math.ceil(train_size * n_samples_y)
        key = jax.random.PRNGKey(seed=seed)
        x = jax.random.permutation(key, x)
        y = jax.random.permutation(key, y)
        if a is not None:
            a = jax.random.permutation(key, a)
        if b is not None:
            b = jax.random.permutation(key, b)
        return (
            x[:n_train_x],
            y[:n_train_y],
            x[n_train_x:],
            y[n_train_y:],
            a[:n_train_x] if a is not None else None,
            b[:n_train_x] if b is not None else None,
            a[n_train_x:] if a is not None else None,
            b[n_train_x:] if b is not None else None,
        )

    @property
    def solver(self) -> NeuralDualSolver:
        """Underlying optimal transport solver."""
        return self._solver

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.LINEAR


class CondNeuralSolver(NeuralSolver):
    """Solver class solving Conditional Neural Optimal Transport problems."""

    def _prepare(  # type: ignore[override]
        self,
        xy: Dict[Any, Tuple[TaggedArray, ArrayLike, ArrayLike]],
        sample_pairs: List[Tuple[Any, Any]],
        train_size: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[JaxSampler, JaxSampler]:
        train_data: List[Optional[ArrayLike]] = []
        train_a: List[Optional[ArrayLike]] = []
        train_b: List[Optional[ArrayLike]] = []
        valid_data: List[Optional[ArrayLike]] = []
        valid_a: List[Optional[ArrayLike]] = []
        valid_b: List[Optional[ArrayLike]] = []

        sample2idx: Dict[int, Any] = {}
        kwargs = _filter_kwargs(JaxSampler, **kwargs)
        if train_size == 1.0:
            train_data = [d[0].data_src for d in xy.values()]
            train_a = [d[1] for d in xy.values()]
            train_b = [d[2] for d in xy.values()]
            valid_data, valid_a, valid_b = train_data, train_a, train_b
            sample2idx = {k: i for i, k in enumerate(xy.keys())}
        else:
            if train_size > 1.0 or train_size <= 0.0:
                raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

            seed = kwargs.pop("seed", 0)
            for i, (k, (d, a, b)) in enumerate(xy.items()):
                t_data, v_data, t_a, t_b, v_a, v_b = self._split_data(
                    d.data_src, train_size=train_size, seed=seed, a=a, b=b
                )
                train_data.append(t_data)
                train_a.append(t_a)
                train_b.append(t_b)
                valid_data.append(v_data)
                valid_a.append(v_a)
                valid_b.append(v_b)
                sample2idx[k] = i

        self._train_sampler = JaxSampler(
            train_data, sample_pairs, a=train_a, b=train_b, sample2idx=sample2idx, **kwargs
        )
        self._valid_sampler = JaxSampler(
            valid_data, sample_pairs, a=valid_a, b=valid_b, sample2idx=sample2idx, **kwargs
        )
        return (self._train_sampler, self._valid_sampler)

    def _split_data(  # type:ignore[override]
        self,
        x: ArrayLike,
        train_size: float,
        seed: int,
        a: Optional[ArrayLike] = None,
        b: Optional[ArrayLike] = None,
    ) -> Tuple[
        ArrayLike, ArrayLike, Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]
    ]:
        n_samples_x = x.shape[0]
        n_train_x = math.ceil(train_size * n_samples_x)
        key = jax.random.PRNGKey(seed=seed)
        x = jax.random.permutation(key, x)
        if a is not None:
            a = jax.random.permutation(key, a)
        if b is not None:
            b = jax.random.permutation(key, b)
        return (
            x[:n_train_x],
            x[n_train_x:],
            a[:n_train_x] if a is not None else None,
            b[:n_train_x] if b is not None else None,
            a[n_train_x:] if a is not None else None,
            b[n_train_x:] if b is not None else None,
        )

import abc
import inspect
import math
import types
from typing import Any, Dict, List, Literal, Mapping, Optional, Set, Tuple, Union

import jax
import jax.numpy as jnp
import scipy.sparse as sp
from ott.geometry import costs, epsilon_scheduler, geometry, pointcloud
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr

from moscot._types import (
    ArrayLike,
    ProblemKind_t,
    QuadInitializer_t,
    SinkhornInitializer_t,
)
from moscot.backends.ott._gap_models import MongeGapSolver
from moscot.backends.ott._jax_data import JaxSampler
from moscot.backends.ott._neuraldual import OTTNeuralDualSolver
from moscot.backends.ott._utils import (
    _filter_kwargs,
    alpha_to_fused_penalty,
    check_shapes,
    ensure_2d,
)
from moscot.backends.ott.output import (
    CondNeuralDualOutput,
    GapNeuralOutput,
    NeuralDualOutput,
    OTTOutput,
)
from moscot.base.solver import OTSolver
from moscot.costs import get_cost
from moscot.utils.tagged_array import TaggedArray

__all__ = ["SinkhornSolver", "GWSolver", "NeuralDualSolver", "CondNeuralDualSolver"]

OTTSolver_t = Union[
    sinkhorn.Sinkhorn,
    sinkhorn_lr.LRSinkhorn,
    gromov_wasserstein.GromovWasserstein,
    gromov_wasserstein_lr.LRGromovWasserstein,
]
OTTProblem_t = Union[linear_problem.LinearProblem, quadratic_problem.QuadraticProblem]
Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]


class OTTJaxSolver(OTSolver[OTTOutput], abc.ABC):
    """Base class for :mod:`ott` solvers :cite:`cuturi2022optimal`.

    Parameters
    ----------
    jit
        Whether to :func:`~jax.jit` the :attr:`solver`.
    """

    def __init__(self, jit: bool = True):
        super().__init__()
        self._solver: Optional[OTTSolver_t] = None
        self._problem: Optional[OTTProblem_t] = None
        self._jit = jit

    def _create_geometry(
        self,
        x: TaggedArray,
        epsilon: Union[float, epsilon_scheduler.Epsilon] = None,
        relative_epsilon: Optional[bool] = None,
        scale_cost: Scale_t = 1.0,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> geometry.Geometry:
        if x.is_point_cloud:
            cost_fn = x.cost
            if cost_fn is None:
                cost_fn = costs.SqEuclidean()
            elif isinstance(cost_fn, str):
                cost_fn = get_cost(cost_fn, backend="ott", **kwargs)
            if not isinstance(cost_fn, costs.CostFn):
                raise TypeError(f"Expected `cost_fn` to be `ott.geometry.costs.CostFn`, found `{type(cost_fn)}`.")

            y = None if x.data_tgt is None else ensure_2d(x.data_tgt, reshape=True)
            x = ensure_2d(x.data_src, reshape=True)
            if y is not None and x.shape[1] != y.shape[1]:
                raise ValueError(
                    f"Expected `x/y` to have the same number of dimensions, found `{x.shape[1]}/{y.shape[1]}`."
                )

            return pointcloud.PointCloud(
                x,
                y=y,
                cost_fn=cost_fn,
                epsilon=epsilon,
                relative_epsilon=relative_epsilon,
                scale_cost=scale_cost,
                batch_size=batch_size,
            )

        arr = ensure_2d(x.data_src, reshape=False)
        if x.is_cost_matrix:
            return geometry.Geometry(
                cost_matrix=arr, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost
            )
        if x.is_kernel:
            return geometry.Geometry(
                kernel_matrix=arr, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost
            )
        raise NotImplementedError(f"Creating geometry from `tag={x.tag!r}` is not yet implemented.")

    def _solve(  # type: ignore[override]
        self,
        prob: OTTProblem_t,
        **kwargs: Any,
    ) -> OTTOutput:
        solver = jax.jit(self.solver) if self._jit else self.solver
        out = solver(prob, **kwargs)
        return OTTOutput(out)

    @property
    def solver(self) -> OTTSolver_t:
        """:mod:`ott` solver."""
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
    """Solver for the :term:`linear problem`.

    The (Kantorovich relaxed) :term:`OT` problem is defined by two distributions in the same space.
    The aim is to obtain a probabilistic map from the source distribution to the target distribution such that
    the (weighted) sum of the distances between coupled data point in the source and the target distribution is
    minimized.

    Parameters
    ----------
    jit
        Whether to :func:`~jax.jit` the :attr:`solver`.
    rank
        Rank of the solver. If `-1`, use :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` :cite:`cuturi:2013`,
        otherwise, use :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn` :cite:`scetbon:21a`.
    epsilon
        Additional epsilon regularization for the low-rank approach.
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
        jit: bool = True,
        rank: int = -1,
        epsilon: float = 0.0,
        initializer: SinkhornInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__(jit=jit)
        if rank > -1:
            kwargs.setdefault("gamma", 10)
            kwargs.setdefault("gamma_rescale", True)
            initializer = "rank2" if initializer is None else initializer
            self._solver = sinkhorn_lr.LRSinkhorn(
                rank=rank, epsilon=epsilon, initializer=initializer, kwargs_init=initializer_kwargs, **kwargs
            )
        else:
            initializer = "default" if initializer is None else initializer
            self._solver = sinkhorn.Sinkhorn(initializer=initializer, kwargs_init=initializer_kwargs, **kwargs)

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        # geometry
        epsilon: Union[float, epsilon_scheduler.Epsilon] = None,
        relative_epsilon: Optional[bool] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        cost_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        cost_matrix_rank: Optional[int] = None,
        # problem
        **kwargs: Any,
    ) -> linear_problem.LinearProblem:
        del x, y
        if xy is None:
            raise ValueError(f"Unable to create geometry from `xy={xy}`.")

        geom = self._create_geometry(
            xy,
            epsilon=epsilon,
            relative_epsilon=relative_epsilon,
            batch_size=batch_size,
            scale_cost=scale_cost,
            **cost_kwargs,
        )
        if cost_matrix_rank is not None:
            geom = geom.to_LRCGeometry(rank=cost_matrix_rank)
        self._problem = linear_problem.LinearProblem(geom, **kwargs)
        return self._problem

    @property
    def xy(self) -> Optional[geometry.Geometry]:
        """Geometry defining the linear term."""
        return None if self._problem is None else self._problem.geom

    @property
    def problem_kind(self) -> ProblemKind_t:  # noqa: D102
        return "linear"

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        geom_kwargs = {"epsilon", "relative_epsilon", "batch_size", "scale_cost", "cost_kwargs", "cost_matrix_rank"}
        problem_kwargs = set(inspect.signature(linear_problem.LinearProblem).parameters.keys())
        problem_kwargs -= {"geom"}
        return geom_kwargs | problem_kwargs, {"epsilon"}


class GWSolver(OTTJaxSolver):
    """Solver for the :term:`quadratic problem` :cite:`memoli:2011`.

    The :term:`Gromov-Wasserstein (GW) <Gromov-Wasserstein>` problem involves two distribution in
    possibly two different spaces. Points in the source distribution are matched to points in the target distribution
    by comparing the relative location of the points within each distribution.

    Parameters
    ----------
    jit
        Whether to :func:`~jax.jit` the :attr:`solver`.
    rank
        Rank of the solver. If `-1` use the full-rank :term:`GW <Gromov-Wasserstein>` :cite:`peyre:2016`,
        otherwise, use the low-rank approach :cite:`scetbon:21b`.
    initializer
        Initializer for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein`.
    initializer_kwargs
        Keyword arguments for the ``initializer``.
    linear_solver_kwargs
        Keyword arguments for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    kwargs
        Keyword arguments for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein` .
    """

    def __init__(
        self,
        jit: bool = True,
        rank: int = -1,
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__(jit=jit)
        if rank > -1:
            kwargs.setdefault("gamma", 10)
            kwargs.setdefault("gamma_rescale", True)
            initializer = "rank2" if initializer is None else initializer
            self._solver = gromov_wasserstein_lr.LRGromovWasserstein(
                rank=rank,
                initializer=initializer,
                kwargs_init=initializer_kwargs,
                **kwargs,
            )
        else:
            linear_ot_solver = sinkhorn.Sinkhorn(**linear_solver_kwargs)
            initializer = None
            self._solver = gromov_wasserstein.GromovWasserstein(
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
        # geometry
        epsilon: Union[float, epsilon_scheduler.Epsilon] = None,
        relative_epsilon: Optional[bool] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        cost_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        cost_matrix_rank: Optional[int] = None,
        # problem
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> quadratic_problem.QuadraticProblem:
        if x is None or y is None:
            raise ValueError(f"Unable to create geometry from `x={x}`, `y={y}`.")
        geom_kwargs: Any = {
            "epsilon": epsilon,
            "relative_epsilon": relative_epsilon,
            "batch_size": batch_size,
            "scale_cost": scale_cost,
            "cost_matrix_rank": cost_matrix_rank,
            **cost_kwargs,
        }
        geom_xx = self._create_geometry(x, **geom_kwargs)
        geom_yy = self._create_geometry(y, **geom_kwargs)
        if alpha == 1.0 or xy is None:  # GW
            # arbitrary fused penalty; must be positive
            geom_xy, fused_penalty = None, 1.0
        else:  # FGW
            fused_penalty = alpha_to_fused_penalty(alpha)
            geom_xy = self._create_geometry(xy, **geom_kwargs)
            check_shapes(geom_xx, geom_yy, geom_xy)

        self._problem = quadratic_problem.QuadraticProblem(
            geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, **kwargs
        )
        return self._problem

    @property
    def x(self) -> Optional[geometry.Geometry]:
        """The first geometry defining the quadratic term."""
        return None if self._problem is None else self._problem.geom_xx

    @property
    def y(self) -> geometry.Geometry:
        """The second geometry defining the quadratic term."""
        return None if self._problem is None else self._problem.geom_yy

    @property
    def xy(self) -> Optional[geometry.Geometry]:
        """Geometry defining the linear term in the :term:`FGW <fused Gromov-Wasserstein>`."""
        return None if self._problem is None else self._problem.geom_xy

    @property
    def is_fused(self) -> Optional[bool]:
        """Whether the solver is fused."""
        return None if self._problem is None else (self.xy is not None)

    @property
    def problem_kind(self) -> ProblemKind_t:  # noqa: D102
        return "quadratic"

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        geom_kwargs = {"epsilon", "relative_epsilon", "batch_size", "scale_cost", "cost_kwargs", "cost_matrix_rank"}
        problem_kwargs = set(inspect.signature(quadratic_problem.QuadraticProblem).parameters.keys())
        problem_kwargs -= {"geom_xx", "geom_yy", "geom_xy", "fused_penalty"}
        problem_kwargs |= {"alpha"}
        return geom_kwargs | problem_kwargs, {"epsilon"}


class NeuralDualSolver(OTSolver[OTTOutput]):
    """Solver class solving Neural Optimal Transport problems."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._train_sampler: Optional[JaxSampler] = None
        self._valid_sampler: Optional[JaxSampler] = None
        kwargs = _filter_kwargs(OTTNeuralDualSolver, **kwargs)
        self._solver = OTTNeuralDualSolver(**kwargs)

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
            [train_x, train_y], policy_pairs=[(0, 1)], a=[train_a, []], b=[[], train_b], **kwargs
        )
        self._valid_sampler = JaxSampler(
            [valid_x, valid_y], policy_pairs=[(0, 1)], a=[valid_a, []], b=[[], valid_b], **kwargs
        )
        return (self._train_sampler, self._valid_sampler)

    def _solve(self, data_samplers: Tuple[JaxSampler, JaxSampler]) -> NeuralDualOutput:  # type: ignore[override]
        model, logs = self.solver(data_samplers[0], data_samplers[1])
        return NeuralDualOutput(model, logs)  # type:ignore[arg-type]

    @staticmethod
    def _assert2d(arr: ArrayLike, *, allow_reshape: bool = True) -> jnp.ndarray:
        arr: jnp.ndarray = jnp.asarray(arr.A if sp.issparse(arr) else arr)  # type: ignore[no-redef, attr-defined, name-defined]   # noqa:E501
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
    def solver(self) -> OTTNeuralDualSolver:
        """Underlying optimal transport solver."""
        return self._solver

    @property
    def problem_kind(self) -> ProblemKind_t:
        """Problem kind."""
        return "linear"

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        return {"trainloader", "validloader"}


class CondNeuralDualSolver(NeuralDualSolver):
    """Solver class solving Conditional Neural Optimal Transport problems."""

    def __init__(self, *args, cond_dim: int, **kwargs: Any) -> None:
        super().__init__(*args, cond_dim=cond_dim, **kwargs)

    def _prepare(  # type: ignore[override]
        self,
        xy: Dict[Any, Tuple[TaggedArray, ArrayLike, ArrayLike]],
        sample_pairs: List[Tuple[Any, Any]],
        train_size: float = 0.9,
        **kwargs: Any,
    ) -> Tuple[JaxSampler, JaxSampler]:
        train_data: List[Optional[ArrayLike]] = []
        train_a: List[Optional[ArrayLike]] = []
        train_b: List[Optional[ArrayLike]] = []
        valid_data: List[Optional[ArrayLike]] = []
        valid_a: List[Optional[ArrayLike]] = []
        valid_b: List[Optional[ArrayLike]] = []

        sample_to_idx: Dict[int, Any] = {}
        kwargs = _filter_kwargs(JaxSampler, **kwargs)
        if train_size == 1.0:
            train_data = [d[0].data_src for d in xy.values()]
            train_a = [d[1] for d in xy.values()]
            train_b = [d[2] for d in xy.values()]
            valid_data, valid_a, valid_b = train_data, train_a, train_b
            sample_to_idx = {k: i for i, k in enumerate(xy.keys())}
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
                sample_to_idx[k] = i

        self._train_sampler = JaxSampler(
            train_data, sample_pairs, conditional=True, a=train_a, b=train_b, sample_to_idx=sample_to_idx, **kwargs
        )
        self._valid_sampler = JaxSampler(
            valid_data, sample_pairs, conditional=True, a=valid_a, b=valid_b, sample_to_idx=sample_to_idx, **kwargs
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

    def _solve(self, data_samplers: Tuple[JaxSampler, JaxSampler]) -> CondNeuralDualOutput:  # type: ignore[override]
        model, logs = self.solver(data_samplers[0], data_samplers[1])
        return CondNeuralDualOutput(output=model, training_logs=logs)  # type:ignore[arg-type]

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        return {"trainloader", "validloader"}


class GapSolver(OTSolver[OTTOutput]):
    """Solver class solving Gap Optimal Transport problems."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._train_sampler: Optional[JaxSampler] = None
        self._valid_sampler: Optional[JaxSampler] = None
        kwargs = _filter_kwargs(MongeGapSolver, **kwargs)
        self._solver = MongeGapSolver(**kwargs)

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
            [train_x, train_y], policy_pairs=[(0, 1)], a=[train_a, []], b=[[], train_b], **kwargs
        )
        self._valid_sampler = JaxSampler(
            [valid_x, valid_y], policy_pairs=[(0, 1)], a=[valid_a, []], b=[[], valid_b], **kwargs
        )
        return (self._train_sampler, self._valid_sampler)

    def _solve(self, data_samplers: Tuple[JaxSampler, JaxSampler]) -> GapNeuralOutput:  # type: ignore[override]
        model, logs = self.solver(data_samplers[0], data_samplers[1])
        return GapNeuralOutput(model, logs)

    @staticmethod
    def _assert2d(arr: ArrayLike, *, allow_reshape: bool = True) -> jnp.ndarray:
        arr: jnp.ndarray = jnp.asarray(arr.A if sp.issparse(arr) else arr)  # type: ignore[no-redef, attr-defined]
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
    def solver(self) -> MongeGapSolver:
        """Underlying optimal transport solver."""
        return self._solver

    @property
    def problem_kind(self) -> ProblemKind_t:
        return "linear"

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        pass

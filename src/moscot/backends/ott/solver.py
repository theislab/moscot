import abc
import inspect
import math
import types
from typing import (
    Any,
    Hashable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.sparse as sp
from ott.geometry import costs, epsilon_scheduler, geometry, pointcloud, geodesic
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr
from ott.neural.flow_models.genot import GENOTLin
from ott.neural.data.datasets import OTDataset, ConditionalOTDataset
from ott.neural.flow_models.models import VelocityField
from ott.neural.flow_models.samplers import uniform_sampler
from ott.neural.models.base_solver import UnbalancednessHandler, OTMatcherLinear
from ott.neural.models.nets import RescalingMLP
from torch.utils.data import DataLoader, RandomSampler
from moscot._types import (
    ArrayLike,
    ProblemKind_t,
    QuadInitializer_t,
    SinkhornInitializer_t,
)
from moscot.backends.ott._utils import alpha_to_fused_penalty, check_shapes, ensure_2d, _instantiate_geodesic_cost
from moscot.backends.ott.output import GraphOTTOutput, OTTOutput
from moscot.backends.ott.output import OTTNeuralOutput, OTTOutput, 
from moscot.base.problems._utils import TimeScalesHeatKernel

from moscot.base.solver import OTSolver
from moscot.costs import get_cost
from moscot.utils.tagged_array import DistributionCollection, TaggedArray

__all__ = ["SinkhornSolver", "GWSolver", "GENOTLinSolver"]

OTTSolver_t = Union[
    sinkhorn.Sinkhorn,
    sinkhorn_lr.LRSinkhorn,
    gromov_wasserstein.GromovWasserstein,
    gromov_wasserstein_lr.LRGromovWasserstein,
]
OTTProblem_t = Union[linear_problem.LinearProblem, quadratic_problem.QuadraticProblem]
Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]
K = TypeVar("K", bound=Hashable)


class SingleDistributionData(NamedTuple):
    data_train: ArrayLike
    data_valid: ArrayLike
    conditions_train: Optional[ArrayLike]
    conditions_valid: Optional[ArrayLike]
    a_train: Optional[ArrayLike]
    a_valid: Optional[ArrayLike]
    b_train: Optional[ArrayLike]
    b_valid: Optional[ArrayLike]


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
        self._a: Optional[jnp.ndarray] = None
        self._b: Optional[jnp.ndarray] = None

    def _create_geometry(
        self,
        x: TaggedArray,
        *,
        is_linear_term: bool,
        epsilon: Union[float, epsilon_scheduler.Epsilon] = None,
        relative_epsilon: Optional[bool] = None,
        scale_cost: Scale_t = 1.0,
        batch_size: Optional[int] = None,
        problem_shape: Optional[Tuple[int, int]] = None,
        t: Optional[float] = None,
        directed: bool = True,
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
        if x.is_graph:  # we currently only support this for the linear term.
            return self._create_graph_geometry(
                is_linear_term=is_linear_term,
                x=x,
                arr=arr,
                problem_shape=problem_shape,
                t=t,
                epsilon=epsilon,
                relative_epsilon=relative_epsilon,
                scale_cost=scale_cost,
                directed=directed,
                **kwargs,
            )
        raise NotImplementedError(f"Creating geometry from `tag={x.tag!r}` is not yet implemented.")

    def _solve(  # type: ignore[override]
        self,
        prob: OTTProblem_t,
        **kwargs: Any,
    ) -> Union[OTTOutput, GraphOTTOutput]:
        solver = jax.jit(self.solver) if self._jit else self.solver
        out = solver(prob, **kwargs)
        if isinstance(prob, linear_problem.LinearProblem) and isinstance(prob.geom, geodesic.Geodesic):
            return GraphOTTOutput(out, shape=(len(self._a), len(self._b)))  # type: ignore[arg-type]
        return OTTOutput(out)

    def _create_graph_geometry(
        self,
        is_linear_term: bool,
        x: TaggedArray,
        arr: jax.Array,
        problem_shape: Optional[Tuple[int, int]],
        t: Optional[float],
        epsilon: Union[float, epsilon_scheduler.Epsilon] = None,
        relative_epsilon: Optional[bool] = None,
        scale_cost: Scale_t = 1.0,
        directed: bool = True,
        **kwargs: Any,
    ) -> geometry.Geometry:
        if x.cost == "geodesic":
            if self.problem_kind == "linear":
                if t is None:
                    if epsilon is None:
                        raise ValueError("`epsilon` cannot be `None`.")
                    return geodesic.Geodesic.from_graph(arr, t=epsilon / 4.0, directed=directed, **kwargs)

                return _instantiate_geodesic_cost(
                    arr=arr,
                    problem_shape=problem_shape,  # type: ignore[arg-type]
                    t=t,
                    is_linear_term=True,
                    epsilon=epsilon,
                    relative_epsilon=relative_epsilon,
                    scale_cost=scale_cost,
                    directed=directed,
                    **kwargs,
                )
            if self.problem_kind == "quadratic":
                problem_shape = x.shape if problem_shape is None else problem_shape
                return _instantiate_geodesic_cost(
                    arr=arr,
                    problem_shape=problem_shape,
                    t=t,
                    is_linear_term=is_linear_term,
                    epsilon=epsilon,
                    relative_epsilon=relative_epsilon,
                    scale_cost=scale_cost,
                    directed=directed,
                    **kwargs,
                )

            raise NotImplementedError(f"Invalid problem kind `{self.problem_kind}`.")
        raise NotImplementedError(f"If the geometry is a graph, `cost` must be `geodesic`, found `{x.cost}`.")

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
        a: jnp.ndarray,
        b: jnp.ndarray,
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
        time_scales_heat_kernel: Optional[TimeScalesHeatKernel] = None,
        # problem
        **kwargs: Any,
    ) -> linear_problem.LinearProblem:
        del x, y
        time_scales_heat_kernel = (
            TimeScalesHeatKernel(None, None, None) if time_scales_heat_kernel is None else time_scales_heat_kernel
        )
        if xy is None:
            raise ValueError(f"Unable to create geometry from `xy={xy}`.")
        self._a = a
        self._b = b
        geom = self._create_geometry(
            xy,
            is_linear_term=True,
            epsilon=epsilon,
            relative_epsilon=relative_epsilon,
            batch_size=batch_size,
            problem_shape=(len(self._a), len(self._b)),
            scale_cost=scale_cost,
            t=time_scales_heat_kernel.xy,
            **cost_kwargs,
        )
        if cost_matrix_rank is not None:
            geom = geom.to_LRCGeometry(rank=cost_matrix_rank)
        if isinstance(geom, geodesic.Geodesic):
            a = jnp.concatenate((a, jnp.zeros_like(self._b)), axis=0)
            b = jnp.concatenate((jnp.zeros_like(self._a), b), axis=0)
        self._problem = linear_problem.LinearProblem(geom, a=a, b=b, **kwargs)
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
        geom_kwargs = {
            "epsilon",
            "relative_epsilon",
            "batch_size",
            "scale_cost",
            "cost_kwargs",
            "cost_matrix_rank",
            "t",
        }
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
        a: jnp.ndarray,
        b: jnp.ndarray,
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
        time_scales_heat_kernel: Optional[TimeScalesHeatKernel] = None,
        # problem
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> quadratic_problem.QuadraticProblem:
        self._a = a
        self._b = b
        time_scales_heat_kernel = (
            TimeScalesHeatKernel(None, None, None) if time_scales_heat_kernel is None else time_scales_heat_kernel
        )
        if x is None or y is None:
            raise ValueError(f"Unable to create geometry from `x={x}`, `y={y}`.")
        geom_kwargs: dict[str, Any] = {
            "epsilon": epsilon,
            "relative_epsilon": relative_epsilon,
            "batch_size": batch_size,
            "scale_cost": scale_cost,
            **cost_kwargs,
        }
        if cost_matrix_rank is not None:
            geom_kwargs["cost_matrix_rank"] = cost_matrix_rank
        geom_xx = self._create_geometry(x, t=time_scales_heat_kernel.x, is_linear_term=False, **geom_kwargs)
        geom_yy = self._create_geometry(y, t=time_scales_heat_kernel.y, is_linear_term=False, **geom_kwargs)
        if alpha == 1.0 or xy is None:  # GW
            # arbitrary fused penalty; must be positive
            geom_xy, fused_penalty = None, 1.0
        else:  # FGW
            fused_penalty = alpha_to_fused_penalty(alpha)
            geom_xy = self._create_geometry(
                xy,
                t=time_scales_heat_kernel.xy,
                problem_shape=(x.shape[0], y.shape[0]),
                is_linear_term=True,
                **geom_kwargs,
            )
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

class GENOTLinSolver(OTSolver[OTTOutput]):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._train_sampler: Optional[ConditionalOTDataset] = None
        self._valid_sampler: Optional[ConditionalOTDataset] = None
        self._neural_kwargs = kwargs

    @property
    def problem_kind(self) -> ProblemKind_t:  # noqa: D102        
        return "linear"  
   
    def _prepare(  # type: ignore[override]
        self,
        distributions: DistributionCollection[K],
        sample_pairs: List[Tuple[Any, Any]],
        train_size: float = 0.9,
        batch_size: int = 1024,
        **kwargs: Any,
    ) -> Tuple[ConditionalOTDataset, ConditionalOTDataset]:
        train_loaders = []
        validate_loaders = []
        if train_size == 1.0:
            for sample_pair in sample_pairs:
                source_key = sample_pair[0]
                target_key = sample_pair[1]
                source_ds = OTDataset(
                    lin=distributions[source_key].xy,
                    conditions=distributions[source_key].conditions,
                )
                source_loader = DataLoader(
                    source_ds,
                    batch_size=batch_size,
                    sampler=RandomSampler(source_ds, replacement=True)
                )
                target_ds = OTDataset(
                    lin=distributions[target_key].xy,
                    conditions=distributions[target_key].conditions
                )
                target_loader = DataLoader(
                    target_ds,
                    batch_size=batch_size,
                    sampler=RandomSampler(target_ds, replacement=True)
                )
                train_loaders.append((source_loader, target_loader))
                validate_loaders.append((source_loader, target_loader))
        else:
            if train_size > 1.0 or train_size <= 0.0:
                raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

            seed = kwargs.pop("seed", 0)
            for sample_pair in sample_pairs:
                source_key = sample_pair[0]
                target_key = sample_pair[1]
                source_split_data = self._split_data(
                    distributions[source_key].xy,
                    conditions=distributions[source_key].conditions,
                    train_size=train_size,
                    seed=seed,
                    a=distributions[source_key].a,
                    b=distributions[source_key].b,
                )
                target_split_data = self._split_data(
                    distributions[target_key].xy,
                    conditions=distributions[target_key].conditions,
                    train_size=train_size,
                    seed=seed,
                    a=distributions[target_key].a,
                    b=distributions[target_key].b,
                )
                source_ds_train = OTDataset(
                    lin=source_split_data.data_train,
                    conditions=source_split_data.conditions_train,
                )
                source_train_loader = DataLoader(
                    source_ds_train,
                    batch_size=batch_size,
                    sampler=RandomSampler(source_ds_train, replacement=True),
                )
                target_ds_train = OTDataset(
                    lin=target_split_data.data_train,
                    conditions=target_split_data.conditions_train
                )
                target_train_loader = DataLoader(
                    target_ds_train,
                    batch_size=batch_size,
                    sampler=RandomSampler(target_ds_train, replacement=True),
                )
                source_ds_validate = OTDataset(
                    lin=source_split_data.data_valid,
                    conditions=source_split_data.conditions_valid,
                )
                source_validate_loader = DataLoader(
                    source_ds_validate,
                    batch_size=batch_size,
                    sampler=RandomSampler(source_ds_validate, replacement=True)
                )
                target_ds_validate = OTDataset(
                    lin=target_split_data.data_valid,
                    conditions=target_split_data.conditions_valid
                )
                target_validate_loader = DataLoader(
                    target_ds_validate,
                    batch_size=batch_size,
                    sampler=RandomSampler(target_ds_validate, replacement=True)
                )
                train_loaders.append((source_train_loader, target_train_loader))
                validate_loaders.append((source_validate_loader, target_validate_loader))
        source_dim = self._neural_kwargs.pop("input_dim")
        target_dim = source_dim
        condition_dim = self._neural_kwargs.pop("cond_dim")
        neural_vf = VelocityField(
            output_dim=target_dim,
            condition_dim=source_dim + condition_dim,
            latent_embed_dim=self._neural_kwargs.pop("latent_embed_dim", 5),
        )
        ot_solver = sinkhorn.Sinkhorn(**self._neural_kwargs.pop("valid_sinkhorn_kwargs", {}))
        tau_a=self._neural_kwargs.pop("tau_a", 1)
        tau_b=self._neural_kwargs.pop("tau_b", 1)
        rescaling_a = self._neural_kwargs.pop("rescaling_a", RescalingMLP(hidden_dim=4, condition_dim=condition_dim))
        rescaling_b = self._neural_kwargs.pop("rescaling_b", RescalingMLP(hidden_dim=4, condition_dim=condition_dim))
        seed = self._neural_kwargs.pop("seed", 0)
        rng = jax.random.PRNGKey(seed)
        ot_matcher = self._neural_kwargs.pop("ot_matcher", OTMatcherLinear(
            ot_solver, tau_a=tau_a, tau_b=tau_b
        ))
        time_sampler = self._neural_kwargs.pop("time_sampler", uniform_sampler)
        unbalancedness_handler = self._neural_kwargs.pop("unbalancedness_handler", UnbalancednessHandler(
            rng=rng, source_dim=source_dim, target_dim=target_dim, cond_dim=condition_dim, tau_a=tau_a, tau_b=tau_b, rescaling_a=rescaling_a, rescaling_b=rescaling_b,
        ))
        optimizer = self._neural_kwargs.pop("optimizer", optax.adam(learning_rate=1e-3))
        self._solver = GENOTLin(
            velocity_field=neural_vf,
            input_dim=source_dim,
            output_dim=target_dim,
            cond_dim=condition_dim,
            ot_matcher=ot_matcher,
            unbalancedness_handler=unbalancedness_handler,
            optimizer=optimizer,
            time_sampler=time_sampler,
            rng=rng,
            matcher_latent_to_data=OTMatcherLinear(sinkhorn.Sinkhorn()) if self._neural_kwargs.pop("solver_latent_to_data", True) else None,
            k_samples_per_x=self._neural_kwargs.pop("k_samples_per_x", 1),
            **self._neural_kwargs
        )
        return ConditionalOTDataset(datasets=train_loaders, seed=seed), ConditionalOTDataset(datasets=validate_loaders, seed=seed)

                
    @staticmethod
    def _assert2d(arr: ArrayLike, *, allow_reshape: bool = True) -> jnp.ndarray:
        arr: jnp.ndarray = jnp.asarray(arr.A if sp.issparse(arr) else arr)  # type: ignore[no-redef, attr-defined]   # noqa:E501
        if allow_reshape and arr.ndim == 1:
            return jnp.reshape(arr, (-1, 1))
        if arr.ndim != 2:
            raise ValueError(f"Expected array to have 2 dimensions, found `{arr.ndim}`.")
        return arr

    def _split_data(  # TODO: adapt for Gromov terms
        self,
        x: ArrayLike,
        conditions: Optional[ArrayLike],
        train_size: float,
        seed: int,
        a: Optional[ArrayLike] = None,
        b: Optional[ArrayLike] = None,
    ) -> SingleDistributionData:
        n_samples_x = x.shape[0]
        n_train_x = math.ceil(train_size * n_samples_x)
        rng = np.random.default_rng(seed)
        x = rng.permutation(x)
        if a is not None:
            a = rng.permutation(a)
        if b is not None:
            b = rng.permutation(b)

        return SingleDistributionData(
            data_train=x[:n_train_x],
            data_valid=x[n_train_x:],
            conditions_train=conditions[:n_train_x] if conditions is not None else None,
            conditions_valid=conditions[n_train_x:] if conditions is not None else None,
            a_train=a[:n_train_x] if a is not None else None,
            a_valid=a[n_train_x:] if a is not None else None,
            b_train=b[:n_train_x] if b is not None else None,
            b_valid=b[n_train_x:] if b is not None else None,
        )

    @property
    def solver(self) -> GENOTLin:
        """Underlying optimal transport solver."""
        return self._solver

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        return {"batch_size", "train_size", "trainloader", "validloader"}, {}  # type: ignore[return-value]

    def _solve(self, data_samplers: Tuple[ConditionalOTDataset, ConditionalOTDataset]) -> OTTNeuralOutput:  # type: ignore[override]
        self.solver(data_samplers[0], data_samplers[1])
        return OTTNeuralOutput(self.solver)

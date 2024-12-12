import abc
import functools
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
    TypeVar,
    Union,
)

import optax

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs, epsilon_scheduler, geodesic, geometry, pointcloud
from ott.neural.datasets import OTData, OTDataset
from ott.neural.methods.flows import dynamics, genot
from ott.neural.networks.layers import time_encoder
from ott.neural.networks.velocity_field import VelocityField
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr
from ott.solvers.utils import uniform_sampler

from moscot._logging import logger
from moscot._types import (
    ArrayLike,
    LRInitializer_t,
    ProblemKind_t,
    QuadInitializer_t,
    SinkhornInitializer_t,
)
from moscot.backends.ott._utils import (
    InitializerResolver,
    Loader,
    MultiLoader,
    _instantiate_geodesic_cost,
    alpha_to_fused_penalty,
    check_shapes,
    convert_scipy_sparse,
    data_match_fn,
    densify,
    ensure_2d,
)
from moscot.backends.ott.output import GraphOTTOutput, OTTNeuralOutput, OTTOutput
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
    initializer_kwargs
        Keyword arguments for the initializer.
    """

    def __init__(self, jit: bool = True, initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({})):
        super().__init__()
        self._solver: Optional[OTTSolver_t] = None
        self._problem: Optional[OTTProblem_t] = None
        self._jit = jit
        self._a: Optional[jnp.ndarray] = None
        self._b: Optional[jnp.ndarray] = None

        self.initializer_kwargs = initializer_kwargs

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

            y = None if x.data_tgt is None else densify(ensure_2d(x.data_tgt, reshape=True))
            x = densify(ensure_2d(x.data_src, reshape=True))
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
        arr = densify(arr) if x.is_graph else convert_scipy_sparse(arr)

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
        out = solver(prob, **self.initializer_kwargs, **kwargs)
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
        super().__init__(jit=jit, initializer_kwargs=initializer_kwargs)
        if rank > -1:
            kwargs.setdefault("gamma", 500)
            kwargs.setdefault("gamma_rescale", True)
            eps = kwargs.get("epsilon")
            if eps is not None and eps > 0.0:
                logger.info(f"Found `epsilon`={eps}>0. We recommend setting `epsilon`=0 for the low-rank solver.")
            if isinstance(initializer, str):
                initializer = InitializerResolver.lr_from_str(initializer, rank=rank)
            self._solver = sinkhorn_lr.LRSinkhorn(rank=rank, epsilon=epsilon, initializer=initializer, **kwargs)
        else:
            if isinstance(initializer, str):
                initializer = InitializerResolver.from_str(initializer)
            self._solver = sinkhorn.Sinkhorn(initializer=initializer, **kwargs)

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
        initializer: QuadInitializer_t | LRInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__(jit=jit, initializer_kwargs=initializer_kwargs)
        if rank > -1:
            kwargs.setdefault("gamma", 10)
            kwargs.setdefault("gamma_rescale", True)
            eps = kwargs.get("epsilon")
            if eps is not None and eps > 0.0:
                logger.info(f"Found `epsilon`={eps}>0. We recommend setting `epsilon`=0 for the low-rank solver.")
            if isinstance(initializer, str):
                initializer = InitializerResolver.lr_from_str(initializer, rank=rank)
            self._solver = gromov_wasserstein_lr.LRGromovWasserstein(
                rank=rank,
                initializer=initializer,
                **kwargs,
            )
        else:
            linear_solver = sinkhorn.Sinkhorn(**linear_solver_kwargs)
            if isinstance(initializer, str):
                raise ValueError("Expected `initializer` to be `None` or `ott.initializers.quadratic.initializers`.")
            self._solver = gromov_wasserstein.GromovWasserstein(
                linear_solver=linear_solver,
                initializer=initializer,
                **kwargs,
            )

    def _prepare(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        alpha: float,
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
        if alpha <= 0.0:
            raise ValueError(f"Expected `alpha` to be in interval `(0, 1]`, found `{alpha}`.")
        if (alpha == 1.0 and xy is not None) or (alpha != 1.0 and xy is None):
            raise ValueError(f"Expected `xy` to be `None` if `alpha` is not 1.0, found xy={xy}, alpha={alpha}.")

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
            geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, a=self._a, b=self._b, **kwargs
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
    """Solver class for genot.GENOT linear :cite:`klein2023generative`."""

    def __init__(self, **kwargs: Any) -> None:
        """Initiate the class with any kwargs passed to the ott-jax class."""
        super().__init__()
        self._train_sampler: Optional[MultiLoader] = None
        self._valid_sampler: Optional[MultiLoader] = None
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
        is_conditional: bool = True,
        **kwargs: Any,
    ) -> Tuple[MultiLoader, MultiLoader]:
        train_loaders = []
        validate_loaders = []
        seed = kwargs.get("seed")
        is_aligned = kwargs.get("is_aligned", False)
        if train_size == 1.0:
            for sample_pair in sample_pairs:
                source_key = sample_pair[0]
                target_key = sample_pair[1]
                src_data = OTData(
                    lin=distributions[source_key].xy,
                    condition=distributions[source_key].conditions if is_conditional else None,
                )
                tgt_data = OTData(
                    lin=distributions[target_key].xy,
                    condition=distributions[target_key].conditions if is_conditional else None,
                )
                dataset = OTDataset(src_data=src_data, tgt_data=tgt_data, seed=seed, is_aligned=is_aligned)
                loader = Loader(dataset, batch_size=batch_size, seed=seed)
                train_loaders.append(loader)
                validate_loaders.append(loader)
        else:
            if train_size > 1.0 or train_size <= 0.0:
                raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

            seed = kwargs.get("seed", 0)
            for sample_pair in sample_pairs:
                source_key = sample_pair[0]
                target_key = sample_pair[1]
                source_data: ArrayLike = distributions[source_key].xy
                target_data: ArrayLike = distributions[target_key].xy
                source_split_data = self._split_data(
                    source_data,
                    conditions=distributions[source_key].conditions,
                    train_size=train_size,
                    seed=seed,
                    a=distributions[source_key].a,
                    b=distributions[source_key].b,
                )
                target_split_data = self._split_data(
                    target_data,
                    conditions=distributions[target_key].conditions,
                    train_size=train_size,
                    seed=seed,
                    a=distributions[target_key].a,
                    b=distributions[target_key].b,
                )
                src_data_train = OTData(
                    lin=source_split_data.data_train,
                    condition=source_split_data.conditions_train if is_conditional else None,
                )
                tgt_data_train = OTData(
                    lin=target_split_data.data_train,
                    condition=target_split_data.conditions_train if is_conditional else None,
                )
                train_dataset = OTDataset(
                    src_data=src_data_train, tgt_data=tgt_data_train, seed=seed, is_aligned=is_aligned
                )
                train_loader = Loader(train_dataset, batch_size=batch_size, seed=seed)
                src_data_validate = OTData(
                    lin=source_split_data.data_valid,
                    condition=source_split_data.conditions_valid if is_conditional else None,
                )
                tgt_data_validate = OTData(
                    lin=target_split_data.data_valid,
                    condition=target_split_data.conditions_valid if is_conditional else None,
                )
                validate_dataset = OTDataset(
                    src_data=src_data_validate, tgt_data=tgt_data_validate, seed=seed, is_aligned=is_aligned
                )
                validate_loader = Loader(validate_dataset, batch_size=batch_size, seed=seed)
                train_loaders.append(train_loader)
                validate_loaders.append(validate_loader)
        source_dim = self._neural_kwargs.get("input_dim", 0)
        target_dim = source_dim
        condition_dim = self._neural_kwargs.get("cond_dim", 0)
        # TODO(ilan-gold): What are reasonable defaults here?
        neural_vf = VelocityField(
            output_dims=[*self._neural_kwargs.get("velocity_field_output_dims", []), target_dim],
            condition_dims=(
                self._neural_kwargs.get("velocity_field_condition_dims", [source_dim + condition_dim])
                if is_conditional
                else None
            ),
            hidden_dims=self._neural_kwargs.get("velocity_field_hidden_dims", [1024, 1024, 1024]),
            time_dims=self._neural_kwargs.get("velocity_field_time_dims", None),
            time_encoder=self._neural_kwargs.get(
                "velocity_field_time_encoder", functools.partial(time_encoder.cyclical_time_encoder, n_freqs=1024)
            ),
        )
        seed = self._neural_kwargs.get("seed", 0)
        rng = jax.random.PRNGKey(seed)
        data_match_fn_kwargs = self._neural_kwargs.get(
            "data_match_fn_kwargs",
            {} if "data_match_fn" in self._neural_kwargs else {"epsilon": 1e-1, "tau_a": 1.0, "tau_b": 1.0},
        )
        time_sampler = self._neural_kwargs.get("time_sampler", uniform_sampler)
        optimizer = self._neural_kwargs.get("optimizer", optax.adam(learning_rate=1e-4))
        self._solver = genot.GENOT(
            vf=neural_vf,
            flow=self._neural_kwargs.get(
                "flow",
                dynamics.ConstantNoiseFlow(0.1),
            ),
            data_match_fn=functools.partial(
                self._neural_kwargs.get("data_match_fn", data_match_fn), typ="lin", **data_match_fn_kwargs
            ),
            source_dim=source_dim,
            target_dim=target_dim,
            condition_dim=condition_dim if is_conditional else None,
            optimizer=optimizer,
            time_sampler=time_sampler,
            rng=rng,
            latent_noise_fn=self._neural_kwargs.get("latent_noise_fn", None),
            **self._neural_kwargs.get("velocity_field_train_state_kwargs", {}),
        )
        return (
            MultiLoader(datasets=train_loaders, seed=seed),
            MultiLoader(datasets=validate_loaders, seed=seed),
        )

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
    def solver(self) -> genot.GENOT:
        """Underlying optimal transport solver."""
        return self._solver

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        return {"batch_size", "train_size", "trainloader", "validloader", "seed"}, {}  # type: ignore[return-value]

    def _solve(self, data_samplers: Tuple[MultiLoader, MultiLoader]) -> OTTNeuralOutput:  # type: ignore[override]
        seed = self._neural_kwargs.get("seed", 0)  # TODO(ilan-gold): unify rng hadnling like OTT tests
        rng = jax.random.PRNGKey(seed)
        logs = self.solver(
            data_samplers[0], n_iters=self._neural_kwargs.get("n_iters", 100), rng=rng
        )  # TODO(ilan-gold): validation and figure out defualts
        return OTTNeuralOutput(self.solver, logs)

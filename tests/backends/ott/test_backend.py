from typing import Optional, Tuple, Type, Union

import pytest

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs
from ott.geometry.geometry import Geometry
from ott.geometry.low_rank import LRCGeometry
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.problems.quadratic import quadratic_problem
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.linear.sinkhorn import solve as sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein

# from ott.solvers.quadratic.gromov_wasserstein import solve as gromov_wasserstein
from ott.solvers.quadratic.gromov_wasserstein_lr import LRGromovWasserstein

from moscot._types import ArrayLike, Device_t
from moscot.backends.ott import GWSolver, SinkhornSolver
from moscot.backends.ott._utils import alpha_to_fused_penalty
from moscot.base.output import BaseSolverOutput
from moscot.base.solver import O, OTSolver
from moscot.utils.tagged_array import Tag, TaggedArray
from tests._utils import ATOL, RTOL, Geom_t
from tests.plotting.conftest import PlotTester, PlotTesterMeta


class TestSinkhorn:
    @pytest.mark.fast()
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps", [None, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, eps: Optional[float], jit: bool):
        fn = jax.jit(sinkhorn) if jit else sinkhorn
        gt = fn(PointCloud(x, epsilon=eps))
        solver = SinkhornSolver(jit=jit)
        assert solver.xy is None
        assert isinstance(solver.solver, Sinkhorn)

        pred = solver(a=None, b=None, xy=(x, x), epsilon=eps)

        assert solver.rank == -1
        assert not solver.is_low_rank
        assert isinstance(solver.xy, Geometry)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("rank", [5, 10])
    @pytest.mark.parametrize("initializer", ["random", "rank2", "k-means"])
    def test_solver_rank(self, y: Geom_t, rank: Optional[int], initializer: str):
        eps = 1e-2
        lr_sinkhorn = LRSinkhorn(rank=rank, initializer=initializer)
        problem = LinearProblem(PointCloud(y, epsilon=eps))
        gt = lr_sinkhorn(problem)

        solver = SinkhornSolver(rank=rank, initializer=initializer)
        assert solver.rank == rank
        assert solver.is_low_rank
        assert solver.xy is None
        assert isinstance(solver.solver, LRSinkhorn)

        pred = solver(a=None, b=None, xy=(y, y), epsilon=eps)

        assert isinstance(solver.xy, PointCloud)
        assert pred.rank == rank
        np.testing.assert_allclose(solver._problem.geom.cost_matrix, problem.geom.cost_matrix, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize(
        ("rank", "cost_fn"), [(2, costs.Euclidean()), (3, costs.SqPNorm(p=1.5)), (5, costs.ElasticL1(0.1))]
    )
    def test_geometry_rank(self, x: Geom_t, rank: int, cost_fn: costs.CostFn):
        eps = 0.05
        geom = PointCloud(x, epsilon=eps, cost_fn=cost_fn).to_LRCGeometry(rank=rank)
        problem = LinearProblem(geom)
        gt = Sinkhorn()(problem)

        solver = SinkhornSolver()
        assert not solver.is_low_rank

        pred = solver(a=None, b=None, xy=TaggedArray(x, cost=cost_fn), epsilon=eps, cost_matrix_rank=rank)

        assert isinstance(solver.xy, LRCGeometry)
        np.testing.assert_allclose(solver._problem.geom.cost_matrix, problem.geom.cost_matrix, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestGW:
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps", [5e-2, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, y: Geom_t, eps: Optional[float], jit: bool):
        thresh = 1e-2
        pc_x, pc_y = PointCloud(x, epsilon=eps), PointCloud(y, epsilon=eps)
        prob = quadratic_problem.QuadraticProblem(pc_x, pc_y)
        sol = GromovWasserstein(epsilon=eps, threshold=thresh)
        solver = jax.jit(sol, static_argnames=["threshold", "epsilon"]) if jit else sol
        gt = solver(prob)

        solver = GWSolver(jit=jit, epsilon=eps, threshold=thresh)
        assert isinstance(solver.solver, GromovWasserstein)
        assert solver.x is None
        assert solver.y is None

        pred = solver(a=None, b=None, x=x, y=y, tags={"x": "point_cloud", "y": "point_cloud"})

        assert solver.is_fused is False
        assert solver.rank == -1
        assert not solver.is_low_rank
        assert isinstance(solver.x, PointCloud)
        assert isinstance(solver.y, PointCloud)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("eps", [5e-1, 1])
    def test_epsilon(self, x_cost: jnp.ndarray, y_cost: jnp.ndarray, eps: Optional[float]) -> None:
        thresh = 1e-3
        problem = QuadraticProblem(
            geom_xx=Geometry(cost_matrix=x_cost, epsilon=eps), geom_yy=Geometry(cost_matrix=y_cost, epsilon=eps)
        )
        gt = GromovWasserstein(epsilon=eps, threshold=thresh)(problem)
        solver = GWSolver(epsilon=eps, threshold=thresh)

        pred = solver(a=None, b=None, x=x_cost, y=y_cost, tags={"x": Tag.COST_MATRIX, "y": Tag.COST_MATRIX})

        assert solver.is_fused is False
        assert pred.rank == -1
        assert solver.rank == -1
        assert isinstance(solver.x, Geometry)
        assert isinstance(solver.y, Geometry)
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("rank", [-1, 7])
    def test_solver_rank(self, x: Geom_t, y: Geom_t, rank: int) -> None:
        thresh, eps = 1e-2, 1e-2
        if rank > -1:
            gt = LRGromovWasserstein(epsilon=eps, rank=rank, threshold=thresh, initializer="rank2")(
                QuadraticProblem(PointCloud(x, epsilon=eps), PointCloud(y, epsilon=eps))
            )

        else:
            gt = GromovWasserstein(epsilon=eps, rank=rank, threshold=thresh)(
                QuadraticProblem(PointCloud(x, epsilon=eps), PointCloud(y, epsilon=eps))
            )

        solver = GWSolver(rank=rank, epsilon=eps, threshold=thresh)
        pred = solver(a=None, b=None, x=x, y=y, tags={"x": "point_cloud", "y": "point_cloud"})

        assert solver.is_fused is False
        assert solver.rank == rank
        assert pred.rank == rank
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestFGW:
    @pytest.mark.parametrize("alpha", [0.25, 0.75])
    @pytest.mark.parametrize("eps", [1e-2, 1e-1, 5e-1])
    def test_matches_ott(self, x: Geom_t, y: Geom_t, xy: Geom_t, eps: Optional[float], alpha: float) -> None:
        thresh = 1e-2
        xx, yy = xy

        ott_solver = GromovWasserstein(epsilon=eps, threshold=thresh)
        problem = quadratic_problem.QuadraticProblem(
            geom_xx=PointCloud(x, epsilon=eps),
            geom_yy=PointCloud(y, epsilon=eps),
            geom_xy=PointCloud(xx, yy, epsilon=eps),
            fused_penalty=alpha_to_fused_penalty(alpha),
        )
        gt = ott_solver(problem)

        solver = GWSolver(epsilon=eps, threshold=thresh)
        assert isinstance(solver.solver, GromovWasserstein)
        assert solver.xy is None

        pred = solver(
            a=None,
            b=None,
            x=x,
            y=y,
            xy=xy,
            alpha=alpha,
            tags={"x": "point_cloud", "y": "point_cloud", "xy": "point_cloud"},
        )

        assert solver.is_fused is True
        assert solver.rank == -1
        assert pred.rank == -1
        assert isinstance(solver.xy, PointCloud)
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.fast()
    @pytest.mark.parametrize("alpha", [0.1, 0.9])
    def test_alpha(self, x: Geom_t, y: Geom_t, xy: Geom_t, alpha: float) -> None:
        thresh, eps = 5e-2, 1e-1
        xx, yy = xy

        ott_solver = GromovWasserstein(epsilon=eps, threshold=thresh)
        problem = quadratic_problem.QuadraticProblem(
            geom_xx=PointCloud(x, epsilon=eps),
            geom_yy=PointCloud(y, epsilon=eps),
            geom_xy=PointCloud(xx, yy, epsilon=eps),
            fused_penalty=alpha_to_fused_penalty(alpha),
        )
        gt = ott_solver(problem)

        solver = GWSolver(epsilon=eps, threshold=thresh)
        pred = solver(
            a=None,
            b=None,
            x=x,
            y=y,
            xy=xy,
            alpha=alpha,
            tags={"x": "point_cloud", "y": "point_cloud", "xy": "point_cloud"},
        )

        assert solver.is_fused is True
        assert not solver.is_low_rank
        assert pred.rank == -1
        assert not pred.is_low_rank
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("eps", [1e-3, 5e-2])
    def test_epsilon(
        self, x_cost: jnp.ndarray, y_cost: jnp.ndarray, xy_cost: jnp.ndarray, eps: Optional[float]
    ) -> None:
        thresh, alpha = 5e-1, 0.66

        problem = QuadraticProblem(
            geom_xx=Geometry(cost_matrix=x_cost, epsilon=eps),
            geom_yy=Geometry(cost_matrix=y_cost, epsilon=eps),
            geom_xy=Geometry(cost_matrix=xy_cost, epsilon=eps),
            fused_penalty=alpha_to_fused_penalty(alpha),
        )
        gt = GromovWasserstein(epsilon=eps, threshold=thresh)(problem)

        solver = GWSolver(epsilon=eps, threshold=thresh)
        pred = solver(
            a=None,
            b=None,
            x=x_cost,
            y=y_cost,
            xy=xy_cost,
            alpha=alpha,
            tags={"x": Tag.COST_MATRIX, "y": Tag.COST_MATRIX, "xy": Tag.COST_MATRIX},
        )

        assert solver.is_fused is True
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestScaleCost:
    @pytest.mark.parametrize("scale_cost", [1.0, 0.5, "mean", "max_cost", "max_norm", "max_bound"])
    def test_scale(self, x: Geom_t, scale_cost: Union[float, str]) -> None:
        eps = 1e-2
        gt = sinkhorn(PointCloud(x, epsilon=eps, scale_cost=scale_cost))

        solver = SinkhornSolver()
        pred = solver(a=None, b=None, xy=(x, x), epsilon=eps, scale_cost=scale_cost)

        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestSolverOutput:
    def test_properties(self, x: ArrayLike, y: ArrayLike) -> None:
        solver = SinkhornSolver()

        out = solver(a=None, b=None, xy=(x, y), epsilon=1e-1)
        a, b = out.a, out.b

        assert isinstance(a, jnp.ndarray)
        assert a.shape == (out.shape[0],)
        assert isinstance(b, jnp.ndarray)
        assert b.shape == (out.shape[1],)

        assert isinstance(out.converged, bool)
        assert isinstance(out.cost, float)
        assert out.cost >= 0
        assert out.shape == (x.shape[0], y.shape[0])

    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("rank", [-1, 5])
    def test_push(
        self,
        x: Geom_t,
        y: Geom_t,
        ab: Tuple[ArrayLike, ArrayLike],
        rank: int,
        batched: bool,
    ) -> None:
        a, _ = ab
        a, ndim = (a, a.shape[1]) if batched else (a[:, 0], None)
        solver = SinkhornSolver(rank=rank)

        out = solver(a=None, b=None, xy=(x, y))
        p = out.push(a, scale_by_marginals=False)

        assert isinstance(out, BaseSolverOutput)
        assert isinstance(p, jnp.ndarray)
        if batched:
            assert p.shape == (out.shape[1], ndim)
        else:
            assert p.shape == (out.shape[1],)

    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("solver_t", [GWSolver])
    def test_pull(
        self,
        x: ArrayLike,
        y: ArrayLike,
        xy: ArrayLike,
        ab: Tuple[ArrayLike, ArrayLike],
        solver_t: Type[OTSolver[O]],
        batched: bool,
    ) -> None:
        _, b = ab
        b, ndim = (b, b.shape[1]) if batched else (b[:, 0], None)
        xx, yy = xy
        solver = solver_t()

        out = solver(a=None, b=None, x=x, y=y, xy=(xx, yy))
        p = out.pull(b, scale_by_marginals=False)

        assert isinstance(out, BaseSolverOutput)
        assert isinstance(p, jnp.ndarray)
        if batched:
            assert p.shape == (out.shape[0], ndim)
        else:
            assert p.shape == (out.shape[0],)

    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("forward", [False, True])
    def test_scale_by_marginals(self, x: Geom_t, ab: Tuple[ArrayLike, ArrayLike], forward: bool, batched: bool) -> None:
        solver = SinkhornSolver()
        a, _ = ab
        z = a if batched else a[:, 0]

        out = solver(a=None, b=None, xy=(x, x))
        p = (out.push if forward else out.pull)(z, scale_by_marginals=True)

        if batched:
            np.testing.assert_allclose(p.sum(axis=0), z.sum(axis=0))
        else:
            np.testing.assert_allclose(p.sum(), z.sum())

    @pytest.mark.parametrize("device", [None, "cpu", "cpu:0", "cpu:1", "explicit"])
    def test_to_device(self, x: Geom_t, device: Optional[Device_t]) -> None:
        # simple integration test
        solver = SinkhornSolver()
        if device == "explicit":
            device = jax.devices()[0]
            _ = solver(a=None, b=None, xy=(x, x), device=device)
        elif device == "cpu:1":
            with pytest.raises(IndexError, match=r"Unable to fetch the device with `id=1`."):
                _ = solver(a=None, b=None, xy=(x, x), device=device)
        else:
            _ = solver(a=None, b=None, xy=(x, x), device=device)


class TestOutputPlotting(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_costs(self, x: Geom_t, y: Geom_t):
        out = GWSolver()(a=None, b=None, x=x, y=y)
        out.plot_costs()

    def test_plot_costs_last(self, x: Geom_t, y: Geom_t):
        out = GWSolver(rank=2)(a=None, b=None, x=x, y=y)
        out.plot_costs(last=3)

    def test_plot_errors_sink(self, x: Geom_t, y: Geom_t):
        out = SinkhornSolver()(a=None, b=None, xy=(x, y))
        out.plot_errors()

    def test_plot_errors_gw(self, x: Geom_t, y: Geom_t):
        out = GWSolver(a=None, b=None, store_inner_errors=True)(x=x, y=y)
        out.plot_errors()

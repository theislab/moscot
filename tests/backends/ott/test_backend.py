from typing import Type, Tuple, Union, Optional

import pytest

from ott.core import LinearProblem
from ott.geometry import Geometry, PointCloud
from ott.core.sinkhorn import sinkhorn, Sinkhorn
from ott.core.sinkhorn_lr import LRSinkhorn
from ott.core.quad_problems import QuadraticProblem
from ott.core.gromov_wasserstein import GromovWasserstein, gromov_wasserstein
import numpy as np
import jax.numpy as jnp

from tests._utils import ATOL, RTOL, Geom_t
from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.backends.ott._output import GWOutput, SinkhornOutput, LRSinkhornOutput
from moscot.solvers._base_solver import BaseSolver
from moscot.solvers._tagged_array import Tag


class TestSinkhorn:
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps", [None, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, eps: Optional[float], jit: bool):
        gt = sinkhorn(PointCloud(x, epsilon=eps), jit=jit)
        pred = SinkhornSolver(jit=jit)(x, x, epsilon=eps)

        assert isinstance(pred, SinkhornOutput)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("rank", [5, 10])
    def test_rank(self, y: Geom_t, rank: Optional[int]):
        eps = 1e-2
        lr_sinkhorn = LRSinkhorn(rank=rank)
        problem = LinearProblem(PointCloud(y, y, epsilon=eps))

        gt = lr_sinkhorn(problem)
        pred = SinkhornSolver(rank=rank)(y, y, epsilon=eps)

        assert isinstance(pred, LRSinkhornOutput)
        assert pred.rank == rank
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("implicit_diff", [False])
    @pytest.mark.parametrize("inner_iterations", [1, 10])
    def test_rank_in_call(self, x: Geom_t, inner_iterations: int, implicit_diff: bool):
        eps = 1e-2

        solver = SinkhornSolver(inner_iterations=inner_iterations)
        assert not solver.is_low_rank

        for rank in (7, 15):
            lr_sinkhorn = LRSinkhorn(rank=rank, inner_iterations=inner_iterations, implicit_diff=implicit_diff)
            problem = LinearProblem(PointCloud(x, x, epsilon=eps))

            gt = lr_sinkhorn(problem)
            pred = solver(x, x, epsilon=eps, rank=rank)

            assert isinstance(pred, LRSinkhornOutput)
            assert pred.rank == rank
            assert not solver.is_low_rank  # we keep the original rank
            np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestGW:
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps", [5e-2, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, y: Geom_t, eps: Optional[float], jit: bool):
        thresh = 1e-2
        gt = gromov_wasserstein(
            PointCloud(x, epsilon=eps), PointCloud(y, epsilon=eps), threshold=thresh, jit=jit, epsilon=eps
        )
        pred = GWSolver(threshold=thresh, jit=jit)(x, y, epsilon=eps)

        assert isinstance(pred, GWOutput)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("eps", [5e-1, 1])
    def test_epsilon(self, x_cost: jnp.ndarray, y_cost: jnp.ndarray, eps: Optional[float]):
        thresh, default_eps = 1e-3, 1e-1
        solver = GWSolver(threshold=thresh, epsilon=default_eps)

        problem = QuadraticProblem(
            geom_xx=Geometry(cost_matrix=x_cost, epsilon=eps), geom_yy=Geometry(cost_matrix=y_cost, epsilon=eps)
        )
        gt = GromovWasserstein(epsilon=eps, threshold=thresh)(problem)
        pred = solver(x_cost, y_cost, epsilon=eps, tags={"x": Tag.COST_MATRIX, "y": Tag.COST_MATRIX})

        assert solver._solver.epsilon == default_eps
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("init_rank", [-1, 2])
    @pytest.mark.parametrize("call_rank", [-1, 7])
    def test_rank(self, x: Geom_t, y: Geom_t, call_rank: int, init_rank: int):
        thresh, eps = 1e-2, 1e-2
        gt = gromov_wasserstein(
            PointCloud(x, epsilon=eps), PointCloud(y, epsilon=eps), rank=call_rank, threshold=thresh, epsilon=eps
        )
        solver = GWSolver(threshold=thresh, rank=init_rank)
        if init_rank > 0:
            assert solver.is_low_rank
            assert isinstance(solver._linear_solver, LRSinkhorn)
            assert solver.rank == init_rank
            assert solver._linear_solver.rank == init_rank
        else:
            assert not solver.is_low_rank
            assert isinstance(solver._linear_solver, Sinkhorn)
            assert solver.rank == -1

        pred = solver(x, y, epsilon=eps, rank=call_rank)

        if init_rank > 0:
            assert solver.is_low_rank
            assert isinstance(solver._linear_solver, LRSinkhorn)
            assert solver.rank == init_rank
            assert solver._linear_solver.rank == init_rank
        else:
            assert not solver.is_low_rank
            assert isinstance(solver._linear_solver, Sinkhorn)
            assert solver.rank == -1
        assert pred.rank == call_rank
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestFGW:
    @pytest.mark.parametrize("alpha", [0.25, 0.75])
    @pytest.mark.parametrize("eps", [1e-2, 1e-1, 5e-1])
    def test_matches_ott(self, x: Geom_t, y: Geom_t, xy: Geom_t, eps: Optional[float], alpha: float):
        thresh = 1e-2
        xx, yy = xy

        gt = gromov_wasserstein(
            geom_xx=PointCloud(x, epsilon=eps),
            geom_yy=PointCloud(y, epsilon=eps),
            geom_xy=PointCloud(xx, yy, epsilon=eps),
            fused_penalty=FGWSolver._alpha_to_fused_penalty(alpha),
            epsilon=eps,
            threshold=thresh,
        )
        pred = FGWSolver(threshold=thresh)(x, y, xx=xx, yy=yy, alpha=alpha, epsilon=eps)

        assert isinstance(pred, GWOutput)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    def test_alpha_in_call(self, x: Geom_t, y: Geom_t, xy: Geom_t):
        thresh, eps = 5e-2, 1e-1
        xx, yy = xy

        solver = FGWSolver(threshold=thresh)

        for alpha in (0.1, 0.9):
            gt = gromov_wasserstein(
                geom_xx=PointCloud(x, epsilon=eps),
                geom_yy=PointCloud(y, epsilon=eps),
                geom_xy=PointCloud(xx, yy, epsilon=eps),
                fused_penalty=FGWSolver._alpha_to_fused_penalty(alpha),
                epsilon=eps,
                threshold=thresh,
            )
            pred = solver(x, y, xx=xx, yy=yy, alpha=alpha, epsilon=eps)

            assert isinstance(pred, GWOutput)
            assert pred.rank == -1
            np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("eps", [1e-3, 5e-2])
    def test_epsilon(self, x_cost: jnp.ndarray, y_cost: jnp.ndarray, xy_cost: jnp.ndarray, eps: Optional[float]):
        thresh, alpha = 5e-1, 0.66
        default_eps = 1e-1
        solver = FGWSolver(threshold=thresh, epsilon=default_eps)

        problem = QuadraticProblem(
            geom_xx=Geometry(cost_matrix=x_cost, epsilon=eps),
            geom_yy=Geometry(cost_matrix=y_cost, epsilon=eps),
            geom_xy=Geometry(cost_matrix=xy_cost, epsilon=eps),
            fused_penalty=FGWSolver._alpha_to_fused_penalty(alpha),
        )
        gt = GromovWasserstein(epsilon=eps, threshold=thresh)(problem)
        pred = solver(
            x_cost,
            y_cost,
            xx=xy_cost,
            alpha=alpha,
            epsilon=eps,
            tags={"x": Tag.COST_MATRIX, "y": Tag.COST_MATRIX, "xx": Tag.COST_MATRIX},
        )

        assert solver._solver.epsilon == default_eps
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestScaleCost:
    @pytest.mark.parametrize("scale_cost", [None, 0.5, "mean", "max_cost", "max_norm", "max_bound"])
    def test_scale(self, x: Geom_t, scale_cost: Optional[Union[float, str]]):
        eps = 1e-2
        gt = sinkhorn(PointCloud(x, epsilon=eps, scale_cost=scale_cost))

        solver = SinkhornSolver()
        pred = solver(x, epsilon=eps, scale_cost=scale_cost)

        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestSolverOutput:
    def test_properties(self, x: Geom_t, y: Geom_t):
        solver = SinkhornSolver()

        out = solver(x, y, epsilon=1e-1)
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
        ab: Tuple[np.ndarray, np.ndarray],
        rank: int,
        batched: bool,
    ):
        a, ndim = (ab[0], ab[0].shape[1]) if batched else (ab[0][:, 0], None)
        solver = SinkhornSolver(rank=rank)

        out = solver(x, y)
        p = out.push(a, scale_by_marginals=False)

        assert isinstance(out, BaseSolverOutput)
        assert isinstance(p, jnp.ndarray)
        if batched:
            assert p.shape == (out.shape[1], ndim)
        else:
            assert p.shape == (out.shape[1],)

    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("solver_t", [GWSolver, FGWSolver])
    def test_pull(
        self,
        x: Geom_t,
        y: Geom_t,
        xy: Geom_t,
        ab: Tuple[np.ndarray, np.ndarray],
        solver_t: Type[BaseSolver],
        batched: bool,
    ):
        b, ndim = (ab[1], ab[1].shape[1]) if batched else (ab[1][:, 0], None)
        xx, yy = xy
        solver = solver_t()

        out = solver(x, y, xx=xx, yy=yy)
        p = out.pull(b, scale_by_marginals=False)

        assert isinstance(out, BaseSolverOutput)
        assert isinstance(p, jnp.ndarray)
        if batched:
            assert p.shape == (out.shape[0], ndim)
        else:
            assert p.shape == (out.shape[0],)

    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("forward", [False, True])
    def test_scale_by_marginals(self, x: Geom_t, ab: Tuple[np.ndarray, np.ndarray], forward: bool, batched: bool):
        solver = SinkhornSolver()
        z = ab[0] if batched else ab[0][:, 0]

        out = solver(x)
        p = (out.push if forward else out.pull)(z, scale_by_marginals=True)

        if batched:
            np.testing.assert_allclose(p.sum(axis=0), z.sum(axis=0))
        else:
            np.testing.assert_allclose(p.sum(), z.sum())

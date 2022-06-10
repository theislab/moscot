from typing import Type, Tuple, Union, Optional

import pytest

from ott.core import LinearProblem
from ott.geometry import Geometry, PointCloud
from ott.core.sinkhorn import sinkhorn
from ott.core.sinkhorn_lr import LRSinkhorn
from ott.core.quad_problems import QuadraticProblem
from ott.core.gromov_wasserstein import GromovWasserstein, gromov_wasserstein
import numpy as np
import jax.numpy as jnp

from tests._utils import ATOL, RTOL, Geom_t
from moscot._types import ArrayLike
from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver  # type: ignore[attr-defined]
from moscot.solvers._output import BaseSolverOutput
from moscot.backends.ott._output import LinearOutput, LRLinearOutput, QuadraticOutput
from moscot.solvers._base_solver import O, OTSolver
from moscot.solvers._tagged_array import Tag


class TestSinkhorn:
    @pytest.mark.fast()
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps", [None, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, eps: Optional[float], jit: bool) -> None:
        gt = sinkhorn(PointCloud(x, epsilon=eps), jit=jit)
        pred = SinkhornSolver(jit=jit)(xy=(x, x), epsilon=eps)

        assert isinstance(pred, LinearOutput)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("rank", [5, 10])
    def test_rank(self, y: Geom_t, rank: Optional[int]) -> None:
        eps = 1e-2
        lr_sinkhorn = LRSinkhorn(rank=rank)
        problem = LinearProblem(PointCloud(y, y, epsilon=eps))

        gt = lr_sinkhorn(problem)
        pred = SinkhornSolver(rank=rank)(xy=(y, y), epsilon=eps)

        assert isinstance(pred, LRLinearOutput)
        assert pred.rank == rank
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestGW:
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps", [5e-2, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, y: Geom_t, eps: Optional[float], jit: bool) -> None:
        thresh = 1e-2
        gt = gromov_wasserstein(
            PointCloud(x, epsilon=eps), PointCloud(y, epsilon=eps), threshold=thresh, jit=jit, epsilon=eps
        )
        pred = GWSolver(threshold=thresh, jit=jit)(x=x, y=y, epsilon=eps)

        assert isinstance(pred, QuadraticOutput)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("eps", [5e-1, 1])
    def test_epsilon(self, x_cost: jnp.ndarray, y_cost: jnp.ndarray, eps: Optional[float]) -> None:
        thresh = 1e-3

        problem = QuadraticProblem(
            geom_xx=Geometry(cost_matrix=x_cost, epsilon=eps), geom_yy=Geometry(cost_matrix=y_cost, epsilon=eps)
        )
        gt = GromovWasserstein(epsilon=eps, threshold=thresh)(problem)
        solver = GWSolver(threshold=thresh)
        pred = solver(x=x_cost, y=y_cost, epsilon=eps, tags={"x": Tag.COST_MATRIX, "y": Tag.COST_MATRIX})

        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("rank", [-1, 7])
    def test_rank(self, x: Geom_t, y: Geom_t, rank: int) -> None:
        thresh, eps = 1e-2, 1e-2
        gt = gromov_wasserstein(
            PointCloud(x, epsilon=eps), PointCloud(y, epsilon=eps), rank=rank, threshold=thresh, epsilon=eps
        )
        solver = GWSolver(threshold=thresh, rank=rank)
        pred = solver(x=x, y=y, epsilon=eps)

        assert pred.rank == rank
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestFGW:
    @pytest.mark.parametrize("alpha", [0.25, 0.75])
    @pytest.mark.parametrize("eps", [1e-2, 1e-1, 5e-1])
    def test_matches_ott(self, x: Geom_t, y: Geom_t, xy: Geom_t, eps: Optional[float], alpha: float) -> None:
        thresh = 1e-2

        gt = gromov_wasserstein(
            geom_xx=PointCloud(x, epsilon=eps),
            geom_yy=PointCloud(y, epsilon=eps),
            geom_xy=PointCloud(xy[0], xy[1], epsilon=eps),
            fused_penalty=FGWSolver._alpha_to_fused_penalty(alpha),
            epsilon=eps,
            threshold=thresh,
        )
        pred = FGWSolver(threshold=thresh)(x=x, y=y, xy=xy, alpha=alpha, epsilon=eps)

        assert isinstance(pred, QuadraticOutput)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.fast()
    @pytest.mark.parametrize("alpha", [0.1, 0.9])
    def test_alpha(self, x: Geom_t, y: Geom_t, xy: Geom_t, alpha: float) -> None:
        thresh, eps = 5e-2, 1e-1

        gt = gromov_wasserstein(
            geom_xx=PointCloud(x, epsilon=eps),
            geom_yy=PointCloud(y, epsilon=eps),
            geom_xy=PointCloud(xy[0], xy[1], epsilon=eps),
            fused_penalty=FGWSolver._alpha_to_fused_penalty(alpha),
            epsilon=eps,
            threshold=thresh,
        )
        solver = FGWSolver(threshold=thresh)
        pred = solver(x=x, y=y, xy=xy, alpha=alpha, epsilon=eps)

        assert isinstance(pred, QuadraticOutput)
        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("eps", [1e-3, 5e-2])
    def test_epsilon(
        self, x_cost: jnp.ndarray, y_cost: jnp.ndarray, xy_cost: jnp.ndarray, eps: Optional[float]
    ) -> None:
        thresh, alpha = 5e-1, 0.66
        solver = FGWSolver(threshold=thresh)

        problem = QuadraticProblem(
            geom_xx=Geometry(cost_matrix=x_cost, epsilon=eps),
            geom_yy=Geometry(cost_matrix=y_cost, epsilon=eps),
            geom_xy=Geometry(cost_matrix=xy_cost, epsilon=eps),
            fused_penalty=FGWSolver._alpha_to_fused_penalty(alpha),
        )
        gt = GromovWasserstein(epsilon=eps, threshold=thresh)(problem)
        pred = solver(
            x=x_cost,
            y=y_cost,
            xy=xy_cost,
            alpha=alpha,
            epsilon=eps,
            tags={"x": Tag.COST_MATRIX, "y": Tag.COST_MATRIX, "xy": Tag.COST_MATRIX},
        )

        assert pred.rank == -1
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestScaleCost:
    @pytest.mark.parametrize("scale_cost", [None, 0.5, "mean", "max_cost", "max_norm", "max_bound"])
    def test_scale(self, x: Geom_t, scale_cost: Optional[Union[float, str]]) -> None:
        eps = 1e-2
        gt = sinkhorn(PointCloud(x, epsilon=eps, scale_cost=scale_cost))

        solver = SinkhornSolver()
        pred = solver(xy=(x, x), epsilon=eps, scale_cost=scale_cost)

        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=RTOL, atol=ATOL)


class TestSolverOutput:
    def test_properties(self, x: ArrayLike, y: ArrayLike) -> None:
        solver = SinkhornSolver()

        out = solver(xy=(x, y), epsilon=1e-1)
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
        a, ndim = (ab[0], ab[0].shape[1]) if batched else (ab[0][:, 0], None)
        solver = SinkhornSolver(rank=rank)

        out = solver(xy=(x, y))
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
        x: ArrayLike,
        y: ArrayLike,
        xy: ArrayLike,
        ab: Tuple[ArrayLike, ArrayLike],
        solver_t: Type[OTSolver[O]],  # noqa: E741
        batched: bool,
    ) -> None:
        b, ndim = (ab[1], ab[1].shape[1]) if batched else (ab[1][:, 0], None)
        xx, yy = xy
        solver = solver_t()

        out = solver(x=x, y=y, xy=(xx, yy))
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
        z = ab[0] if batched else ab[0][:, 0]

        out = solver(xy=(x, x))
        p = (out.push if forward else out.pull)(z, scale_by_marginals=True)

        if batched:
            np.testing.assert_allclose(p.sum(axis=0), z.sum(axis=0))
        else:
            np.testing.assert_allclose(p.sum(), z.sum())

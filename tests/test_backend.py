from typing import Optional

from conftest import Geom_t
import pytest

from ott.core import LinearProblem
from ott.geometry import PointCloud
from ott.core.sinkhorn import sinkhorn
from ott.core.sinkhorn_lr import LRSinkhorn
from ott.core.gromov_wasserstein import gromov_wasserstein
import numpy as np

from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.backends.ott._output import GWOutput, SinkhornOutput, LRSinkhornOutput

_RTOL = 1e-6
_ATOL = 1e-6


class TestSinkhorn:
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps,", [None, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, eps: Optional[float], jit: bool):
        gt = sinkhorn(PointCloud(x, epsilon=eps), jit=jit)
        pred = SinkhornSolver(jit=jit)(x, x, eps=eps)

        assert isinstance(pred, SinkhornOutput)
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=_RTOL, atol=_ATOL)

    @pytest.mark.parametrize("rank", [5, 10])
    def test_rank(self, y: Geom_t, rank: Optional[int]):
        eps = 1e-2
        lr_sinkhorn = LRSinkhorn(rank=rank)
        problem = LinearProblem(PointCloud(y, y, epsilon=eps))

        gt = lr_sinkhorn(problem)
        pred = SinkhornSolver(rank=rank)(y, y, eps=eps)

        assert isinstance(pred, LRSinkhornOutput)
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=_RTOL, atol=_ATOL)

    @pytest.mark.parametrize("inner_iterations", [1, 10])
    def test_rank_in_call(self, x: Geom_t, inner_iterations: int):
        eps = 1e-2

        solver = SinkhornSolver(inner_iterations=inner_iterations)
        assert not solver.is_low_rank

        for rank in (7, 15):
            lr_sinkhorn = LRSinkhorn(rank=rank, inner_iterations=inner_iterations)
            problem = LinearProblem(PointCloud(x, x, epsilon=eps))

            gt = lr_sinkhorn(problem)
            pred = solver(x, x, eps=eps, rank=rank)

            assert isinstance(pred, LRSinkhornOutput)
            assert solver.rank == rank
            np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=_RTOL, atol=_ATOL)


class TestGW:
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps,", [None, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, y: Geom_t, eps: Optional[float], jit: bool):
        gt = gromov_wasserstein(PointCloud(x, epsilon=eps), PointCloud(y, epsilon=eps))
        pred = GWSolver()(x, y, eps=eps)

        assert isinstance(pred, GWOutput)
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=_RTOL, atol=_ATOL)


class TestFGW:
    @pytest.mark.parametrize("alpha", [0.5, 1.5])
    @pytest.mark.parametrize("eps,", [None, 1e-2, 1e-1])
    def test_matches_ott(self, x: Geom_t, y: Geom_t, xy: Geom_t, eps: Optional[float], alpha: float):
        xx, yy = xy

        gt = gromov_wasserstein(
            geom_xx=PointCloud(x, epsilon=eps),
            geom_yy=PointCloud(y, epsilon=eps),
            geom_xy=PointCloud(xx, yy, epsilon=eps),
            fused_penalty=alpha,
        )
        pred = FGWSolver()(x, y, xx=xx, yy=yy, alpha=alpha, eps=eps)

        assert isinstance(pred, GWOutput)
        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=_RTOL, atol=_ATOL)

    def test_alpha_in_call(self, x: Geom_t, y: Geom_t, xy: Geom_t):
        eps = 1e-1
        xx, yy = xy

        solver = FGWSolver()
        assert not solver.is_low_rank

        for alpha in (0.1, 0.9):
            gt = gromov_wasserstein(
                geom_xx=PointCloud(x, epsilon=eps),
                geom_yy=PointCloud(y, epsilon=eps),
                geom_xy=PointCloud(xx, yy, epsilon=eps),
                fused_penalty=alpha,
            )
            pred = solver(x, y, xx=xx, yy=yy, alpha=alpha, eps=eps)

            assert isinstance(pred, GWOutput)
            np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=_RTOL, atol=_ATOL)

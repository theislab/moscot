from typing import Optional

from conftest import Geom_t
import pytest

from ott.geometry import PointCloud
from ott.core.sinkhorn import sinkhorn
import numpy as np

from moscot.backends.ott import SinkhornSolver

_RTOL = 1e-6
_ATOL = 1e-6


class TestSinkhorn:
    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("eps,", [None, 1e-2, 1e-1])
    def test_matches_ott(self, geom_xx: Geom_t, eps: Optional[float], jit: bool):
        x, y = geom_xx
        gt = sinkhorn(PointCloud(x, y, epsilon=eps), jit=jit)
        pred = SinkhornSolver(jit=jit)(x, y, eps=eps)

        np.testing.assert_allclose(gt.matrix, pred.transport_matrix, rtol=_RTOL, atol=_ATOL)

    def test_rank(self):
        pass

    def test_eps(self):
        pass

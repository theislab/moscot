from typing import Type, Tuple, Union, Optional

import pytest

from ott.geometry.geometry import Geometry
from ott.geometry.low_rank import LRCGeometry
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear.sinkhorn import sinkhorn, Sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from ott.problems.linear.linear_problem import LinearProblem
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein, gromov_wasserstein
import jax
import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from tests._utils import ATOL, RTOL, Geom_t, MockSolverOutput
from moscot._types import Device_t, ArrayLike
from moscot.backends.ott import GWSolver, SinkhornSolver  # type: ignore[attr-defined]
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import O, OTSolver
from moscot.solvers._tagged_array import Tag





class TestBaseSolverOutput:
    @pytest.mark.parametrize("batch_size", [1,4])
    @pytest.mark.parametrize("threshold", [0.0, 1e-8, 1.0])
    def test_sparsify(self, batch_size: int, threshold: float) -> None:
        tmap = np.abs(np.random.rand(49,30))
        output=MockSolverOutput(tmap/tmap.sum())
        res = output.sparsify(threshold=threshold, batch_size=batch_size)
        assert isinstance(res, csr_matrix)
        assert res.shape == (49,30)
        assert np.all(res.data >= 0) 
        if threshold == 0.0:
            assert np.all(res.data == 0)
        if threshold == 1e-8:
            assert np.all(res.data == 0 + res.data >= 1e-8)
        if threshold == 1.0:
            np.testing.assert_allclose(np.asarray(res), tmap, rtol=RTOL, atol=ATOL)
            
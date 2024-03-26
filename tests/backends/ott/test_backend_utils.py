import pytest

import jax.experimental.sparse as jesp
import numpy as np
import scipy.sparse as sp
from ott.geometry.geometry import Geometry

from moscot.backends.ott._utils import _instantiate_geodesic_cost


class TestBackendUtils:

    @staticmethod
    def test_instantiate_geodesic_cost():
        m, n = 10, 10
        problem_shape = 10, 10
        g = sp.rand(m, n, 0.1, dtype=np.float64)
        g = jesp.BCOO.from_scipy_sparse(g)
        geom = _instantiate_geodesic_cost(g, problem_shape, 1.0, False)
        assert isinstance(geom, Geometry)
        with pytest.raises(ValueError, match="Expected `x` to have"):
            _instantiate_geodesic_cost(g, problem_shape, 1.0, True)
        geom = _instantiate_geodesic_cost(g, (5, 5), 1.0, True)

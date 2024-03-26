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
from ott.solvers.linear import solve as sinkhorn
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
from ott.solvers.quadratic.gromov_wasserstein_lr import LRGromovWasserstein

from moscot._types import ArrayLike, Device_t
from moscot.backends.ott import GWSolver, SinkhornSolver
from moscot.backends.ott._utils import alpha_to_fused_penalty
from moscot.base.output import BaseSolverOutput
from moscot.base.solver import O, OTSolver
from moscot.backends.ott._utils import _instantiate_geodesic_cost

import jax.experimental.sparse as jesp
import scipy.sparse as sp
import networkx as nx
from networkx.generators import balanced_tree, random_graphs


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








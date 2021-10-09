from typing import Type, Optional

import pytest

from jax import numpy as jnp
from ott.core.sinkhorn import sinkhorn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import gromov_wasserstein
import numpy as np

from moscot import GW, FusedGW, Regularized
from moscot._base import BaseSolver
from moscot._solver import BaseGW


@pytest.mark.parametrize("solver_t", [Regularized, GW, FusedGW])
def test_solver_runs(geom_a: Geometry, geom_b: Geometry, geom_ab: Geometry, solver_t: Type[BaseSolver]):
    solver = solver_t()

    with pytest.raises(RuntimeError, match=r"Not fitted\."):
        _ = solver.matrix

    if isinstance(solver, Regularized):
        solver.fit(geom_a)
    elif isinstance(solver, GW):
        solver.fit(geom_a, geom_b)
    elif isinstance(solver, FusedGW):
        solver.fit(geom_a, geom_b, geom_ab)

    assert isinstance(solver.matrix, jnp.ndarray)
    assert isinstance(solver.converged, bool)
    if isinstance(solver, BaseGW):
        assert isinstance(solver.converged_sinkhorn, list)


def test_sinkhorn_matches_jax(geom_a: Geometry):
    solver = Regularized()

    solver = solver.fit(geom_a)
    res = sinkhorn(geom_a)
    transport = geom_a.transport_from_potentials(res.f, res.g)

    np.testing.assert_allclose(solver.matrix, transport, rtol=1e-5)


@pytest.mark.parametrize("jit", [False, True])
def test_gw_matches_jax(geom_a: Geometry, geom_b: Geometry, jit: bool):
    solver = GW(jit=jit, epsilon=0.01)

    solver = solver.fit(geom_a, geom_b)
    res = gromov_wasserstein(geom_a, geom_b, sinkhorn_kwargs=solver._kwargs, jit=jit, epsilon=solver.epsilon)

    np.testing.assert_allclose(solver.matrix, res.transport, rtol=1e-5)


@pytest.mark.parametrize("rtol", [-1.0, 1.0])
def test_fgw_converged(geom_a: Geometry, geom_b: Geometry, geom_ab: Geometry, rtol: float):
    max_iters = 10
    solver = FusedGW().fit(geom_a, geom_b, geom_ab, rtol=rtol, atol=-1.0, max_iterations=max_iters)

    if rtol == 1.0:
        assert solver.converged
        assert len(solver.converged_sinkhorn) == 1
    elif rtol == -1.0:
        assert not solver.converged
        assert len(solver.converged_sinkhorn) == max_iters


@pytest.mark.parametrize("eps", [None, 1e-2, 1e-3])
def test_regularized_eps(geom_ab: Geometry, eps: Optional[float]):
    solver = Regularized(epsilon=eps)
    solver.fit(geom_ab)

    if eps is None:
        eps = 0.05
    assert geom_ab.epsilon == eps

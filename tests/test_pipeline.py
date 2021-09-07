from typing import Type

import pytest

from jax import numpy as jnp
from ott.core.sinkhorn import sinkhorn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import gromov_wasserstein
import numpy as np

from scott import UnbalancedOT, GromowWassersteinOT, FusedGromowWassersteinOT
from scott._base import BaseSolver


@pytest.mark.parametrize("solver_t", [UnbalancedOT, GromowWassersteinOT, FusedGromowWassersteinOT])
def test_solver_runs(geom_a: Geometry, geom_b: Geometry, geom_ab: Geometry, solver_t: Type[BaseSolver]):
    solver = solver_t()

    with pytest.raises(RuntimeError, match=r"Not fitted\."):
        _ = solver.matrix

    if isinstance(solver, UnbalancedOT):
        solver.fit(geom_a)
    elif isinstance(solver, GromowWassersteinOT):
        solver.fit(geom_a, geom_b)
    elif isinstance(solver, FusedGromowWassersteinOT):
        solver.fit(geom_a, geom_b, geom_ab)

    assert isinstance(solver.matrix, jnp.ndarray)


def test_sinkhorn_matches_jax(geom_a: Geometry):
    solver = UnbalancedOT()

    solver = solver.fit(geom_a)
    res = sinkhorn(geom_a)
    transport = geom_a.transport_from_potentials(res.f, res.g)

    np.testing.assert_allclose(solver.matrix, transport, rtol=1e-5)


def test_gw_matches_jax(geom_a: Geometry, geom_b: Geometry):
    solver = GromowWassersteinOT()

    solver = solver.fit(geom_a, geom_b)
    res = gromov_wasserstein(geom_a, geom_b, sinkhorn_kwargs={})

    np.testing.assert_allclose(solver.matrix, res.transport, rtol=1e-5)

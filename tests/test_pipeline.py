from typing import Type, Optional

from conftest import create_marginals
from pytest_mock import MockerFixture
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
    solver = GW(jit=jit)

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
    eps_orig = geom_ab.epsilon
    solver = Regularized(epsilon=eps)
    solver.fit(geom_ab)

    if eps is None:
        eps = eps_orig
    assert geom_ab.epsilon == eps_orig
    assert solver._transport.geom.epsilon == eps


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("eps", [None, 1e-2])
def test_random_init_coupling_epsilon(eps: Optional[float], uniform: bool):
    a, b = create_marginals(32, 64, uniform=uniform, seed=42)
    solver = FusedGW(epsilon=eps)
    if eps is None:
        with pytest.raises(ValueError, match=r"Please specify `epsilon="):
            _ = solver._get_initial_coupling(a, b, method="random")
        return

    T = solver._get_initial_coupling(a, b, method="random")

    assert isinstance(T, jnp.ndarray)
    np.testing.assert_array_equal(T.shape, (len(a), len(b)))
    np.testing.assert_allclose(T.sum(1), a, rtol=1e-6)
    np.testing.assert_allclose(T.sum(0), b, rtol=1e-6)


@pytest.mark.parametrize("uniform", [False, True])
def test_random_init_coupling_reproducible(uniform: bool):
    a, b = create_marginals(32, 64, uniform=uniform, seed=42)

    T1 = FusedGW(epsilon=1e-2)._get_initial_coupling(a, b, method="random", seed=42)
    T2 = FusedGW(epsilon=1e-2)._get_initial_coupling(a, b, method="random", seed=42)
    T3 = FusedGW(epsilon=1e-2)._get_initial_coupling(a, b, method="random", seed=0)

    np.testing.assert_allclose(T1, T2)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(T1, T3)


def test_fgw_not_converged_warns(geom_a: Geometry, geom_b: Geometry, geom_ab: Geometry):
    solver = FusedGW(epsilon=1e-3)

    with pytest.warns(UserWarning, match=r"Maximum number of iterations \(1\) reached"):
        try:
            solver.fit(geom_a, geom_b, geom_ab, rtol=1e-12, atol=1e-12, max_iterations=1)
        except ValueError:
            # in case marginals are not satisfied
            pass


@pytest.mark.parametrize("mismatch", [False, True])
def test_marginals_check(geom_a: Geometry, geom_b: Geometry, mocker: MockerFixture, mismatch: bool):
    a, b = create_marginals(geom_a.shape[0], geom_b.shape[0], uniform=False, seed=42)
    tmat = jnp.outer(a, b) + float(mismatch)

    solver = GW()
    mocker.patch.object(solver, "_matrix", new=tmat)

    if mismatch:
        with pytest.raises(ValueError, match=r"\nNot equal to tolerance"):
            solver._check_marginals(a, b)
    else:
        solver._check_marginals(a, b)

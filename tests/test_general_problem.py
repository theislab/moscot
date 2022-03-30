from typing import Type

from _utils import TestSolverOutput
<<<<<<< HEAD
from conftest import ATOL, RTOL, Geom_t
=======
from conftest import Geom_t
>>>>>>> origin/tests/time
import pytest

from ott.geometry import PointCloud
from ott.core.sinkhorn import sinkhorn
import numpy as np
import jax.numpy as jnp

from anndata import AnnData

from moscot.problems import GeneralProblem
from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import BaseSolver
<<<<<<< HEAD
from moscot.solvers._tagged_array import Tag
=======
>>>>>>> origin/tests/time


class TestGeneralProblem:
    @pytest.mark.parametrize("solver_t", [SinkhornSolver, GWSolver, FGWSolver])
    def test_simple_run(self, adata_x: AnnData, adata_y: AnnData, solver_t: Type[BaseSolver]):
        prob = GeneralProblem(adata_x, adata_y, solver=solver_t())
        prob = prob.prepare(
            x={"attr": "X"}, y={"attr": "X"}, xy={"x_attr": "obsm", "x_key": "X_pc", "y_attr": "obsm", "y_key": "X_pc"}
        ).solve(epsilon=5e-1)
        sol = prob.solution

        assert isinstance(sol, BaseSolverOutput)

    def test_set_solver(self, adata_x: AnnData):
        prob = GeneralProblem(adata_x, solver=SinkhornSolver())
        assert isinstance(prob.solver, SinkhornSolver)

        prob.solver = FGWSolver()
        assert isinstance(prob.solver, FGWSolver)

    def test_output(self, adata_x: AnnData, x: Geom_t):
        problem = GeneralProblem(adata_x)
        problem._solution = TestSolverOutput(x * x.T)

        assert problem.solution.shape == (len(x), len(x))

    @pytest.mark.parametrize("scale_cost", ["max_cost", "max_bound"])
    def test_passing_scale(self, adata_x: AnnData, scale_cost: str):
        scale_cost, online, eps = "max_cost", True, 5e-2
        gt = sinkhorn(PointCloud(jnp.asarray(adata_x.X), online=online, epsilon=eps, scale_cost=scale_cost))

        prob = GeneralProblem(adata_x, solver=SinkhornSolver())
        prob = prob.prepare(x={"attr": "X", "tag": Tag.POINT_CLOUD}).solve(
            online=online, epsilon=eps, scale_cost=scale_cost
        )
        sol = prob.solution

        np.testing.assert_allclose(gt.matrix, sol.transport_matrix, rtol=RTOL, atol=ATOL)


class MultiMarginalProblem:
    pass

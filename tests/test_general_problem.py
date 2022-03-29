from typing import Type

from conftest import ATOL, RTOL
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
from moscot.solvers._tagged_array import Tag


class TestGeneralProblem:
    @pytest.mark.parametrize("solver_t", [SinkhornSolver, GWSolver, FGWSolver])
    def test_simple_run(self, adata_x: AnnData, adata_y: AnnData, adata_xy: AnnData, solver_t: Type[BaseSolver]):
        prob = GeneralProblem(adata_x, adata_y, adata_xy=adata_xy, solver=solver_t())
        prob = prob.prepare(x={"attr": "X"}, y={"attr": "X"}, xy={"attr": "X", "tag": "cost"}).solve(epsilon=5e-1)
        sol = prob.solution

        assert isinstance(sol, BaseSolverOutput)

    def test_set_solver(self, adata_x: AnnData):
        prob = GeneralProblem(adata_x, solver=SinkhornSolver())
        assert isinstance(prob.solver, SinkhornSolver)

        prob.solver = FGWSolver()
        assert isinstance(prob.solver, FGWSolver)

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

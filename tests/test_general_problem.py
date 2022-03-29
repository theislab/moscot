from typing import Type

import pytest

from anndata import AnnData

from moscot.problems import GeneralProblem
from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import BaseSolver


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


class MultiMarginalProblem:
    pass

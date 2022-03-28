from typing import Type, Optional, Tuple, List

from pytest_mock import MockerFixture
from sklearn.metrics.pairwise import euclidean_distances
import pytest

from anndata import AnnData
import numpy as np

from moscot.problems import SingleCompoundProblem
from moscot.problems.time._lineage import TemporalProblem, TemporalBaseProblem
from moscot.backends.ott import FGWSolver, SinkhornSolver
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.solvers._tagged_array import Tag, TaggedArray
from moscot.problems._base_problem import BaseProblem, GeneralProblem
from moscot.problems._compound_problem import Callback_t
from moscot.solvers._output import BaseSolverOutput

from tests.test_backend import _RTOL, _ATOL


class TestTemporalProblem:

    @pytest.mark.paramterize("growth_genes", [(["gene_1", "gene_2"], ["gene_3", "gene_4"]), (["gene_1", "gene_2"], None)])
    def test_score_genes_for_marginals(self, adata_time: AnnData, growth_genes: Tuple[Optional[List], Optional[List]]):
        problem = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem.score_genes_for_marginals(gene_set_proliferation=growth_genes[0], gene_set_apoptosis=growth_genes[1])

        assert problem._proliferation_key is not None
        assert problem._apoptosis_key is None if growth_genes[1] is None else not None


    def test_prepare(self, adata_time: AnnData):
        expected_keys = {("0", "1"), ("1", "2")}
        problem = TemporalProblem(adata=adata_time, solver=SinkhornSolver())

        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solution is None

        problem = problem.prepare(
            time_key = "time",
            axis="obs",
            policy="sequential",
        )
        
        for key, subprob in problem:
            assert isinstance(subprob, TemporalBaseProblem)
            assert key in expected_keys

    @pytest.mark.parametrize("eps", [1e1, 1e-1])
    def test_solve_balanced(self, adata_time: AnnData, eps: float):
        expected_keys = {("0", "1"), ("1", "2")}
        problem = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem = problem.prepare(
            "time"
        )
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solution:
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    @pytest.mark.parametrize("taus", [9e-1, 1e-2])
    def test_solve_unbalanced(self, adata_time: AnnData, taus: float):
        problem1 = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem2 = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem1 = problem1.prepare(
            "time"
        )
        problem2 = problem2.prepare(
            "time"
        )
        problem1 = problem1.solve(tau_a=taus[0], tau_b=taus[0])
        problem2 = problem2.solve(tau_a=taus[1], tau_b=taus[1])

        assert problem1[0,1].a is not None
        assert problem1[0,1].b is not None
        assert problem2[0,1].a is not None
        assert problem2[0,1].b is not None

        div1 = np.linalg.norm(problem1.solution[0,1].a[:,-1] - np.ones(len(problem1.solution[0,1].a[:,-1]))/len(problem1.solution[0,1].a[:,-1]))
        div2 = np.linalg.norm(problem2.solution[0,1].a[:,-1] - np.ones(len(problem2.solution[0,1].a[:,-1]))/len(problem2.solution[0,1].a[:,-1]))
        assert div1 <= div2

    @pytest.mark.parametrize("n_iters", [3]) #TODO(@MUCDK) as soon as @michalk8 unified warnings/errors test for negative value
    def test_multiple_iterations(self, adata_time: AnnData, n_iters: int):
        problem = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem = problem.prepare("time")
        problem = problem.solve(n_iters=n_iters)
        
        assert problem[0,1].growth_rates.shape[1] == n_iters+1
        assert problem[0,1].growth_rates[:,0] == np.ones(len(problem.solution[0,1].a[:,-1]))/len(problem.solution[0,1].a[:,-1])
        np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, problem[0,1].growth_rates[:,0], problem[0,1].growth_rates[:,1])

    


    


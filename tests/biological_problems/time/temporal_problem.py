from typing import Type, Optional, Tuple

from pytest_mock import MockerFixture
from sklearn.metrics.pairwise import euclidean_distances
import pytest

from anndata import AnnData

from moscot.problems import SingleCompoundProblem
from moscot.problems.time._lineage import TemporalProblem, TemporalBaseProblem
from moscot.backends.ott import FGWSolver, SinkhornSolver
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.solvers._tagged_array import Tag, TaggedArray
from moscot.problems._base_problem import BaseProblem, GeneralProblem
from moscot.problems._compound_problem import Callback_t
from moscot.solvers._output import BaseSolverOutput



class TestTemporalProblem():

    @pytest.mark.paramterize("growth_genes", [(["gene_1", "gene_2"], ["gene_3", "gene_4"]), (["gene_1", "gene_2"], None)])
    def test_score_genes_for_marginals(self, adata_time: AnnData, growth_genes: Tuple[Optional[str], Optional[str]]):
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

    


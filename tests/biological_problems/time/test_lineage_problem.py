from typing import List, Optional, Tuple

import pytest

import numpy as np

from anndata import AnnData

from moscot.backends.ott import FGWSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._lineage import LineageProblem, TemporalBaseProblem


class TestTemporalProblem:
    @pytest.mark.paramterize(
        "growth_genes", [(["gene_1", "gene_2"], ["gene_3", "gene_4"]), (["gene_1", "gene_2"], None)]
    )
    def test_score_genes_for_marginals(self, adata_time: AnnData, growth_genes: Tuple[Optional[List], Optional[List]]): #TODO(@MUCDK) add test once we added default genes 
        problem = LineageProblem(adata=adata_time, solver=FGWSolver())
        problem.score_genes_for_marginals(gene_set_proliferation=growth_genes[0], gene_set_apoptosis=growth_genes[1])

        assert problem._proliferation_key is not None
        assert problem._apoptosis_key is None if growth_genes[1] is None else not None

    def test_prepare_with_barcodes(self, adata_time_barcodes: AnnData):
        expected_keys = {(0, 1), (1, 2)}
        problem = LineageProblem(adata=adata_time_barcodes, solver=FGWSolver())
        problem = problem.prepare(
            time_key="time",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "point_cloud"},
            axis="obs",
            policy="sequential",
        )

        for key, subprob in problem:
            assert isinstance(subprob, TemporalBaseProblem)
            assert key in expected_keys

    def test_prepare_with_trees(self, adata_time_trees: AnnData): #TODO(@MUCDK) create
        pass

    
    @pytest.mark.parametrize(
        "n_iters", [3]
    )  # TODO(@MUCDK) as soon as @michalk8 unified warnings/errors test for negative value
    def test_multiple_iterations(self, adata_time: AnnData, n_iters: int):
        problem = LineageProblem(adata=adata_time, solver=FGWSolver())
        problem = problem.prepare("time")
        problem = problem.solve(n_iters=n_iters)

        assert problem[0, 1].growth_rates.shape[1] == n_iters + 1
        assert problem[0, 1].growth_rates[:, 0] == np.ones(len(problem.solution[0, 1].a[:, -1])) / len(
            problem.solution[0, 1].a[:, -1]
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            problem[0, 1].growth_rates[:, 0],
            problem[0, 1].growth_rates[:, 1],
        )
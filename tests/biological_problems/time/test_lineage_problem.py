import pytest

import numpy as np

from anndata import AnnData

from moscot.backends.ott import FGWSolver
from moscot.problems.time._lineage import LineageProblem, TemporalBaseProblem


class TestTemporalProblem:
    # TODO(@MUCDK) add test for estimate_marginals

    def test_pipeline_with_barcodes(self, adata_time_barcodes: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_barcodes, solver=FGWSolver())
        problem = problem.prepare(
            time_key="time",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost", "loss": "barcode_distance"},
            axis="obs",
            policy="sequential",
        )
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], TemporalBaseProblem)

    def test_pipeline_with_custom_cost(self, adata_time_custom_cost_xy: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_custom_cost_xy, solver=FGWSolver())
        problem = problem.prepare(time_key="time")
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], TemporalBaseProblem)

    def test_pipeline_with_trees(self, adata_time_trees: AnnData):  # TODO(@MUCDK) create
        pass

    @pytest.mark.parametrize(
        "n_iters", [3]
    )  # TODO(@MUCDK) as soon as @michalk8 unified warnings/errors test for negative value
    def test_multiple_iterations(self, adata_time_custom_cost_xy: AnnData, n_iters: int):
        problem = LineageProblem(adata=adata_time_custom_cost_xy, solver=FGWSolver())
        problem = problem.prepare("time")
        problem = problem.solve(n_iters=n_iters)

        assert problem[0, 1].growth_rates.shape[1] == n_iters + 1
        assert all(
            problem[0, 1].growth_rates[:, 0] == np.ones(len(problem[0, 1].a[:, -1])) / len(problem[0, 1].a[:, -1])
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            problem[0, 1].growth_rates[:, 0],
            problem[0, 1].growth_rates[:, 1],
        )

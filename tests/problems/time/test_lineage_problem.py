import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.time import LineageProblem
from moscot.problems.base._birth_death import BirthDeathBaseProblem


class TestLineageProblem:
    @pytest.mark.fast()
    def test_barcodes_pipeline(self, adata_time_barcodes: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_barcodes)
        problem = problem.prepare(
            time_key="time",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost", "loss": "barcode_distance"},
            axis="obs",
            policy="sequential",
        )
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], problem._base_problem_type)

    def test_custom_cost_pipeline(self, adata_time_custom_cost_xy: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_custom_cost_xy)
        problem = problem.prepare(time_key="time")
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathBaseProblem)

    def test_trees_pipeline(self, adata_time_trees: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_trees)
        problem = problem.prepare(time_key="time", lineage_attr={"attr": "uns", "tag": "cost", "loss": "leaf_distance"})
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathBaseProblem)

    @pytest.mark.parametrize(
        "n_iters", [3]
    )  # TODO(@MUCDK) as soon as @michalk8 unified warnings/errors test for negative value
    def test_multiple_iterations_pipeline(self, adata_time_custom_cost_xy: AnnData, n_iters: int):
        problem = LineageProblem(adata=adata_time_custom_cost_xy)
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

    def test_cell_costs_pipeline(self, adata_time_custom_cost_xy: AnnData):
        problem = LineageProblem(adata=adata_time_custom_cost_xy)
        problem = problem.prepare("time")
        problem = problem.solve()

        assert problem.cell_costs_source is None
        assert problem.cell_costs_target is None

import pytest

from anndata import AnnData

from moscot.problems.time import LineageProblem
from moscot.problems.base._birth_death import BirthDeathProblem


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
            assert isinstance(problem[key], BirthDeathProblem)

    @pytest.mark.skip(
        reason="Disabled passing info which cell distribution OTProblem belongs to. Hence, cannot pass trees in `uns`."
    )
    def test_trees_pipeline(self, adata_time_trees: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_trees)
        problem = problem.prepare(time_key="time", lineage_attr={"attr": "uns", "tag": "cost", "loss": "leaf_distance"})
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    def test_cell_costs_pipeline(self, adata_time_custom_cost_xy: AnnData):
        problem = LineageProblem(adata=adata_time_custom_cost_xy)
        problem = problem.prepare("time")
        problem = problem.solve()

        assert problem.cell_costs_source is None
        assert problem.cell_costs_target is None

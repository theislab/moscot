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

    def test_pass_arguments(self, adata_time: AnnData):
        problem = LineageProblem(adata=adata_time)

        problem = problem.prepare(
            time_key="time",
            policy="sequential",
            filter=[(0, 1)],
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost", "loss": "barcode_distance"},
        )

        args_to_check = {
            "alpha": 0.4,
            "epsilon": 0.7,
            "tau_a": 1.0,
            "tau_b": 1.0,
            "scale_cost": "max_cost",
            "rank": 7,
            "batch_size": 123,
            "initializer": "rank2",
            "initializer_kwargs": {},
            "jit": False,
            "threshold": 2e-3,
            "lse_mode": True,
            "norm_error": 2,
            "inner_iterations": 3,
            "min_iterations": 4,
            "max_iterations": 9,
            "gamma": 9.4,
            "gamma_rescale": False,
        }

        solver_args = {
            "lse_mode": "lse_mode",
            "threshold": "threshold",
            "norm_error": "norm_error",
            "inner_iterations": "inner_iterations",
            "min_iterations": "min_iterations",
            "max_iterations": "max_iterations",
            "initializer": "quad_initializer",
            "initializer_kwargs": "kwargs_init",
            "jit": "jit",
            "warm_start": "_warm_start",
        }
        quad_prob_args = {
            "epsilon": "epsilon",
            "scale_cost": "scale_cost",
            "batch_size": "batch_size",
            "tau_a": "_tau_a",
            "tau_b": "_tau_b",
        }
        geometry_args = {"epsilon": "_epsilon_init", "scale_cost": "scale_cost"}
        pointcloud_args = {
            "cost": "cost_fn",
            "power": "power",
            "batch_size": "_batch_size",
            "_scale_cost": "scale_cost",
        }

        problem = problem.solve(**args_to_check)

        solver = problem[(0, 1)]._solver._solver
        for arg in solver_args:
            assert hasattr(solver, solver_args[arg])
            assert getattr(solver, arg) == args_to_check[solver_args[arg]]

        quad_prob = problem[(0, 1)]._solver._problem
        for arg in quad_prob_args:
            assert hasattr(quad_prob, quad_prob_args[arg])
            assert getattr(quad_prob, arg) == args_to_check[quad_prob_args[arg]]
        assert hasattr(quad_prob, "alpha")
        assert getattr(quad_prob, "alpha") == solver._alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom
        for arg in geometry_args:
            assert hasattr(geom, geometry_args[arg])
            assert getattr(geom, arg) == args_to_check[geometry_args[arg]]

        quad_prob.geom
        for arg in pointcloud_args:
            assert hasattr(geom, geometry_args[arg])
            assert getattr(geom, arg) == args_to_check[geometry_args[arg]]


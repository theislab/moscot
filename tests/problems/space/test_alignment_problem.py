from typing import Any, Literal, Mapping, Optional
from pathlib import Path

import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.space import AlignmentProblem
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_solver_args,
    quad_prob_args,
    pointcloud_args,
    gw_linear_solver_args,
)

# TODO(giovp): refactor as fixture
SOLUTIONS_PATH = Path("./tests/data/alignment_solutions.pkl")  # base is moscot


class TestAlignmentProblem:
    @pytest.mark.fast()
    @pytest.mark.parametrize(
        "joint_attr", [{"x_attr": "X", "y_attr": "X"}]
    )  # TODO(giovp): check that callback is correct
    def test_prepare_sequential(self, adata_space_rotate: AnnData, joint_attr: Optional[Mapping[str, Any]]):
        n_obs = adata_space_rotate.shape[0] // 3  # adata is made of 3 datasets
        n_var = adata_space_rotate.shape[1]
        expected_keys = {("0", "1"), ("1", "2")}
        ap = AlignmentProblem(adata=adata_space_rotate)
        assert len(ap) == 0
        assert ap.problems == {}
        assert ap.solutions == {}

        ap = ap.prepare(batch_key="batch", joint_attr=joint_attr)
        assert len(ap) == 2

        for prob_key in expected_keys:
            assert isinstance(ap[prob_key], ap._base_problem_type)
            assert ap[prob_key].shape == (n_obs, n_obs)
            assert ap[prob_key].x.data_src.shape == ap[prob_key].y.data_src.shape == (n_obs, 2)
            assert ap[prob_key].xy.data_src.shape == ap[prob_key].xy.data_tgt.shape == (n_obs, n_var)

    @pytest.mark.fast()
    @pytest.mark.parametrize("reference", ["0", "1", "2"])
    def test_prepare_star(self, adata_space_rotate: AnnData, reference: str):
        ap = AlignmentProblem(adata=adata_space_rotate)
        assert len(ap) == 0
        assert ap.problems == {}
        assert ap.solutions == {}
        ap = ap.prepare(batch_key="batch", policy="star", reference=reference)
        for prob_key in ap:
            _, ref = prob_key
            assert ref == reference
            assert isinstance(ap[prob_key], ap._base_problem_type)

    @pytest.mark.parametrize(
        ("epsilon", "alpha", "rank", "initializer"),
        [(1, 0.9, -1, None), (1, 0.5, 10, "random"), (1, 0.5, 10, "rank2"), (0.1, 0.1, -1, None)],
    )
    def test_solve_balanced(
        self,
        adata_space_rotate: AnnData,
        epsilon: float,
        alpha: float,
        rank: int,
        initializer: Optional[Literal["random", "rank2"]],
    ):
        kwargs = {}
        if rank > -1:
            kwargs["initializer"] = initializer
            if initializer == "random":
                # kwargs["kwargs_init"] = {"key": 0}
                # kwargs["key"] = 0
                return 0  # TODO(@MUCDK) fix after refactoring
        ap = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch")
            .solve(epsilon=epsilon, alpha=alpha, rank=rank, **kwargs)
        )
        for prob_key in ap:
            assert ap[prob_key].solution.rank == rank
            if initializer != "random":  # TODO: is this valid?
                assert ap[prob_key].solution.converged

        assert np.allclose(*(sol.cost for sol in ap.solutions.values()))
        assert np.all([sol.converged for sol in ap.solutions.values()])
        assert np.all([np.all(~np.isnan(sol.transport_matrix)) for sol in ap.solutions.values()])

    def test_solve_unbalanced(self, adata_space_rotate: AnnData):
        tau_a, tau_b = [0.8, 1]
        marg_a = "a"
        marg_b = "b"
        adata_space_rotate.obs[marg_a] = adata_space_rotate.obs[marg_b] = np.ones(300)
        ap = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch", a=marg_a, b=marg_b)
            .solve(tau_a=tau_a, tau_b=tau_b)
        )
        assert np.all([sol.a is not None for sol in ap.solutions.values()])
        assert np.all([sol.b is not None for sol in ap.solutions.values()])
        assert np.all([sol.converged for sol in ap.solutions.values()])
        assert np.allclose(*(sol.cost for sol in ap.solutions.values()), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_space_rotate: AnnData, args_to_check: Mapping[str, Any]):
        adata_space_rotate = adata_space_rotate[adata_space_rotate.obs["batch"].isin(("0", "1"))]
        key = ("0", "1")
        problem = AlignmentProblem(adata=adata_space_rotate)
        problem = problem.prepare(batch_key="batch", joint_attr={"x_attr": "X", "y_attr": "X"})
        problem = problem.solve(**args_to_check)

        solver = problem[key].solver.solver
        for arg, val in gw_solver_args.items():
            assert hasattr(solver, val)
            assert getattr(solver, val) == args_to_check[arg]

        sinkhorn_solver = solver.linear_ot_solver
        for arg, val in gw_linear_solver_args.items():
            assert hasattr(sinkhorn_solver, val)
            el = (
                getattr(sinkhorn_solver, val)[0]
                if isinstance(getattr(sinkhorn_solver, val), tuple)
                else getattr(sinkhorn_solver, val)
            )
            assert el == args_to_check["linear_solver_kwargs"][arg]

        quad_prob = problem[key]._solver._problem
        for arg, val in quad_prob_args.items():
            assert hasattr(quad_prob, val)
            assert getattr(quad_prob, val) == args_to_check[arg]
        assert hasattr(quad_prob, "fused_penalty")
        assert quad_prob.fused_penalty == problem[key]._solver._alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom_xx
        for arg, val in geometry_args.items():
            assert hasattr(geom, val)
            assert getattr(geom, val) == args_to_check[arg]

        geom = quad_prob.geom_xy
        for arg, val in pointcloud_args.items():
            assert hasattr(geom, val)
            assert getattr(geom, val) == args_to_check[arg]

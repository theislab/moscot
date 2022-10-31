from typing import Any, List, Literal, Mapping, Optional
from pathlib import Path

import pytest

import numpy as np

from anndata import AnnData

from tests._utils import _adata_spatial_split
from moscot.problems.space import MappingProblem
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_solver_args,
    quad_prob_args,
    pointcloud_args,
    gw_linear_solver_args,
)
from moscot.solvers._base_solver import ProblemKind

SOLUTIONS_PATH = Path("./tests/data/mapping_solutions.pkl")  # base is moscot


class TestMappingProblem:
    @pytest.mark.fast()
    @pytest.mark.parametrize("sc_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize(
        "joint_attr", [None, "X_pca", {"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"}]
    )
    def test_prepare(self, adata_mapping: AnnData, sc_attr: Mapping[str, str], joint_attr: Optional[Mapping[str, str]]):
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        expected_keys = {(i, "ref") for i in adatasp.obs.batch.cat.categories}
        n_obs = adataref.shape[0]
        x_n_var = adatasp.obsm["spatial"].shape[1]
        y_n_var = adataref.shape[1] if sc_attr["attr"] == "X" else adataref.obsm["X_pca"].shape[1]
        xy_n_vars = adatasp.X.shape[1] if joint_attr == "default" else adataref.obsm["X_pca"].shape[1]
        mp = MappingProblem(adataref, adatasp)
        assert mp.problems == {}
        assert mp.solutions == {}

        mp = mp.prepare(batch_key="batch", sc_attr=sc_attr, joint_attr=joint_attr)

        assert len(mp) == len(expected_keys)
        for prob_key in expected_keys:
            assert isinstance(mp[prob_key], mp._base_problem_type)
            assert mp[prob_key].shape == (n_obs, n_obs)
            assert mp[prob_key].x.data_src.shape == (n_obs, x_n_var)
            assert mp[prob_key].y.data_src.shape == (n_obs, y_n_var)
            assert mp[prob_key].xy.data_src.shape == mp[prob_key].xy.data_tgt.shape == (n_obs, xy_n_vars)

        # test dummy
        prob_key = ("src", "tgt")
        mp = mp.prepare(sc_attr=sc_attr)
        assert len(mp) == 1
        assert isinstance(mp[prob_key], mp._base_problem_type)
        assert mp[prob_key].shape == (2 * n_obs, n_obs)
        np.testing.assert_array_equal(mp._policy._cat, prob_key)

    @pytest.mark.fast()
    @pytest.mark.parametrize("var_names", ["0", [], [str(i) for i in range(20)]])
    def test_prepare_varnames(self, adata_mapping: AnnData, var_names: Optional[List[str]]):
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        problem_kind = ProblemKind.QUAD_FUSED if len(var_names) else ProblemKind.QUAD
        n_obs = adataref.shape[0]
        x_n_var = adatasp.obsm["spatial"].shape[1]
        y_n_var = adataref.shape[1] if not len(var_names) else len(var_names)

        mp = MappingProblem(adataref, adatasp).prepare(batch_key="batch", sc_attr={"attr": "X"}, var_names=var_names)
        for prob in mp.problems.values():
            assert prob._problem_kind == problem_kind
            assert prob.shape == (n_obs, n_obs)
            assert prob.x.data_src.shape == (n_obs, x_n_var)
            assert prob.y.data_src.shape == (n_obs, y_n_var)

    @pytest.mark.parametrize(
        ("epsilon", "alpha", "rank", "initializer"),
        [(1e-2, 0.9, -1, None), (2, 0.5, 10, "random"), (2, 0.5, 10, "rank2"), (2, 0.1, -1, None)],
    )
    @pytest.mark.parametrize("sc_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize("var_names", [None, [], [str(i) for i in range(20)]])
    def test_solve_balanced(
        self,
        adata_mapping: AnnData,
        epsilon: float,
        alpha: float,
        rank: int,
        sc_attr: Mapping[str, str],
        var_names: Optional[List[str]],
        initializer: Optional[Literal["random", "rank2"]],
    ):
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        kwargs = {}
        if rank > -1:
            kwargs["initializer"] = initializer
            if initializer == "random":
                # kwargs["kwargs_init"] = {"key": 0}
                # kwargs["key"] = 0
                return 0  # TODO(@MUCDK) fix after refactoring
        mp = MappingProblem(adataref, adatasp)
        mp = mp.prepare(batch_key="batch", sc_attr=sc_attr, var_names=var_names)
        mp = mp.solve(epsilon=epsilon, alpha=alpha, rank=rank, **kwargs)

        for prob_key in mp:
            assert mp[prob_key].solution.rank == rank
            if initializer != "random":  # TODO: is this valid?
                assert mp[prob_key].solution.converged

        assert np.allclose(*(sol.cost for sol in mp.solutions.values()))
        assert np.all([sol.converged for sol in mp.solutions.values()])
        assert np.all([np.all(~np.isnan(sol.transport_matrix)) for sol in mp.solutions.values()])

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_mapping: AnnData, args_to_check: Mapping[str, Any]):
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        problem = MappingProblem(adataref, adatasp)

        adatasp = adatasp[adatasp.obs["batch"] == 1]

        key = ("1", "ref")
        problem = problem.prepare(batch_key="batch", sc_attr={"attr": "obsm", "key": "X_pca"})
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

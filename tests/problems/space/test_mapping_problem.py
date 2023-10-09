from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional

import pytest

import numpy as np
from ott.geometry import epsilon_scheduler

from anndata import AnnData

from moscot.backends.ott._utils import alpha_to_fused_penalty
from moscot.problems.space import MappingProblem
from tests._utils import _adata_spatial_split
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_linear_solver_args,
    gw_lr_linear_solver_args,
    gw_solver_args,
    pointcloud_args,
    quad_prob_args,
)

# TODO(michalk8): should be made relative to tests root
SOLUTIONS_PATH = Path("./tests/data/mapping_solutions.pkl")  # base is moscot


class TestMappingProblem:
    @pytest.mark.fast()
    @pytest.mark.parametrize("sc_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize("joint_attr", [None, "X_pca", {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize("normalize_spatial", [True, False])
    def test_prepare(
        self,
        adata_mapping: AnnData,
        sc_attr: Mapping[str, str],
        joint_attr: Optional[Mapping[str, str]],
        normalize_spatial: bool,
    ):
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        expected_keys = {(i, "ref") for i in adatasp.obs.batch.cat.categories}
        n_obs = adataref.shape[0]
        x_n_var = adatasp.obsm["spatial"].shape[1]
        y_n_var = adataref.shape[1] if sc_attr["attr"] == "X" else adataref.obsm["X_pca"].shape[1]
        xy_n_vars = adatasp.X.shape[1] if joint_attr == "default" else adataref.obsm["X_pca"].shape[1]
        mp = MappingProblem(adataref, adatasp)
        assert mp.problems == {}
        assert mp.solutions == {}

        mp = mp.prepare(batch_key="batch", sc_attr=sc_attr, joint_attr=joint_attr, normalize_spatial=normalize_spatial)
        if normalize_spatial:
            np.testing.assert_allclose(mp[("1", "ref")].x.data_src.std(), mp[("2", "ref")].x.data_src.std(), atol=1e-15)
            np.testing.assert_allclose(mp[("1", "ref")].x.data_src.std(), 1.0, atol=1e-15)
            np.testing.assert_allclose(mp[("1", "ref")].x.data_src.mean(), 0, atol=1e-15)
            np.testing.assert_allclose(mp[("2", "ref")].x.data_src.mean(), 0, atol=1e-15)

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
        n_obs = adataref.shape[0]
        x_n_var = adatasp.obsm["spatial"].shape[1]
        y_n_var = adataref.shape[1] if not len(var_names) else len(var_names)

        mp = MappingProblem(adataref, adatasp).prepare(batch_key="batch", sc_attr={"attr": "X"}, var_names=var_names)
        for prob in mp.problems.values():
            assert prob.problem_kind == "quadratic"
            # this should hold after running `mp.solve()`
            # assert prob.solver.is_fused is bool(var_names)
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
                return  # TODO(@MUCDK) fix after refactoring
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

        key = ("1", "ref")
        problem = problem.prepare(batch_key="batch", sc_attr={"attr": "obsm", "key": "X_pca"})
        problem = problem.solve(**args_to_check)

        solver = problem[key].solver.solver
        for arg, val in gw_solver_args.items():
            assert hasattr(solver, val)
            assert getattr(solver, val) == args_to_check[arg]

        sinkhorn_solver = solver.linear_ot_solver if args_to_check["rank"] == -1 else solver
        lin_solver_args = gw_linear_solver_args if args_to_check["rank"] == -1 else gw_lr_linear_solver_args
        tmp_dict = args_to_check["linear_solver_kwargs"] if args_to_check["rank"] == -1 else args_to_check
        for arg, val in lin_solver_args.items():
            el = (
                getattr(sinkhorn_solver, val)[0]
                if isinstance(getattr(sinkhorn_solver, val), tuple)
                else getattr(sinkhorn_solver, val)
            )
            assert el == tmp_dict[arg], arg


        quad_prob = problem[key]._solver._problem
        for arg, val in quad_prob_args.items():
            assert hasattr(quad_prob, val)
            assert getattr(quad_prob, val) == args_to_check[arg]
        assert hasattr(quad_prob, "fused_penalty")
        assert quad_prob.fused_penalty == alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom_xx
        for arg, val in geometry_args.items():
            assert hasattr(geom, val)
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            if arg == "epsilon":
                eps_processed = getattr(geom, val)
                assert isinstance(eps_processed, epsilon_scheduler.Epsilon)
                assert eps_processed.target == args_to_check[arg], arg
            else:
                assert getattr(geom, val) == args_to_check[arg], arg
                assert el == args_to_check[arg]

        geom = quad_prob.geom_xy
        for arg, val in pointcloud_args.items():
            assert hasattr(geom, val)
            assert getattr(geom, val) == args_to_check[arg]

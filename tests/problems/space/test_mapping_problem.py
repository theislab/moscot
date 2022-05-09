from typing import Dict, List, Tuple, Optional
from pathlib import Path

import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.space import MappingProblem
from moscot.solvers._base_solver import ProblemKind
from moscot.problems._base_problem import OTProblem

SOLUTIONS_PATH = Path("./../../data/mapping_solutions.pkl")


class TestMappingProblem:
    @staticmethod
    def _adata_split(adata: AnnData) -> Tuple[AnnData, AnnData]:
        adataref = adata[adata.obs.batch == "0"].copy()
        adataref.obsm.pop("spatial")
        adatasp = adata[adata.obs.batch != "0"].copy()
        return adataref, adatasp

    @pytest.mark.parametrize("sc_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize(
        "joint_attr", [None, "default", {"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"}]
    )
    def test_prepare(self, adata_mapping: AnnData, sc_attr: Dict[str, str], joint_attr: Optional[Dict[str, str]]):
        adataref, adatasp = TestMappingProblem._adata_split(adata_mapping)
        expected_keys = [(i, "ref") for i in adatasp.obs.batch.cat.categories]
        n_obs = adataref.shape[0]
        x_n_var = adatasp.obsm["spatial"].shape[1]
        y_n_var = adataref.shape[1] if sc_attr["attr"] == "X" else adataref.obsm["X_pca"].shape[1]
        xy_n_vars = adatasp.X.shape[1] if joint_attr == "default" else adataref.obsm["X_pca"].shape[1]
        problem = MappingProblem(adataref, adatasp)
        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solutions is None

        if joint_attr == "default":
            problem = problem.prepare(batch_key="batch", sc_attr=sc_attr)
        else:
            problem = problem.prepare(batch_key="batch", sc_attr=sc_attr, joint_attr=joint_attr)
        for prob_key, exp_key in zip(problem, expected_keys):
            assert prob_key == exp_key
            assert isinstance(problem[prob_key], OTProblem)
            assert problem[prob_key].shape == (n_obs, n_obs)
            assert problem[prob_key].x.data.shape == (n_obs, x_n_var)
            assert problem[prob_key].y.data.shape == (n_obs, y_n_var)
            assert problem[prob_key].xy[0].data.shape == problem[prob_key].xy[1].data.shape == (n_obs, xy_n_vars)

    @pytest.mark.parametrize("var_names", ["0", [], [str(i) for i in range(20)]])
    def test_prepare_varnames(self, adata_mapping: AnnData, var_names: Optional[List[str]]):
        adataref, adatasp = TestMappingProblem._adata_split(adata_mapping)
        problem_kind = (ProblemKind.QUAD_FUSED if len(var_names) else ProblemKind.QUAD).value
        n_obs = adataref.shape[0]
        x_n_var = adatasp.obsm["spatial"].shape[1]
        y_n_var = adataref.shape[1] if not len(var_names) else len(var_names)

        problem = MappingProblem(adataref, adatasp).prepare(
            batch_key="batch", sc_attr={"attr": "X"}, var_names=var_names
        )
        for prob in problem.problems.values():
            assert prob._problem_kind.value == problem_kind
            assert prob.shape == (n_obs, n_obs)
            assert prob.x.data.shape == (n_obs, x_n_var)
            assert prob.y.data.shape == (n_obs, y_n_var)

    @pytest.mark.parametrize(
        ("epsilon", "alpha", "rank"),
        [
            (1e-2, 0.9, None),
            (2, 0.5, None),
        ],  # TODO(giovp): rank doesn't work?
    )
    @pytest.mark.parametrize("sc_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize("var_names", [None, [], [str(i) for i in range(20)]])
    def test_solve_balance(
        self,
        adata_mapping: AnnData,
        epsilon: float,
        alpha: float,
        rank: int,
        sc_attr: Dict[str, str],
        var_names: Optional[List[Optional[str]]],
    ):
        adataref, adatasp = TestMappingProblem._adata_split(adata_mapping)
        problem = (
            MappingProblem(adataref, adatasp)
            .prepare(batch_key="batch", sc_attr=sc_attr, var_names=var_names)
            .solve(epsilon=epsilon, alpha=alpha, rank=rank)
        )

        epsilon = 1.0 if epsilon is None else epsilon
        False if rank is None else True
        rank = -1 if rank is None else rank
        for prob_key in problem:
            assert problem[prob_key].solution.rank == rank
            # assert problem[prob_key].solution.converged ## never converges

        assert np.allclose(*(sol.cost for sol in problem.solutions.values()))
        # assert np.all([sol.converged for sol in problem.solutions.values()]) ## never converges
        assert np.all([np.all(~np.isnan(sol.transport_matrix)) for sol in problem.solutions.values()])

    @pytest.mark.parametrize("sc_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize("var_names", ["0", [str(i) for i in range(20)]])
    def test_analysis(
        self,
        adata_mapping: AnnData,
        sc_attr: Dict[str, str],
        var_names: Optional[List[Optional[str]]],
    ):
        adataref, adatasp = TestMappingProblem._adata_split(adata_mapping)
        problem = MappingProblem(adataref, adatasp).prepare(batch_key="batch", sc_attr=sc_attr).solve()

        corr = problem.correlate(var_names)
        imp = problem.impute()
        pd.testing.assert_series_equal(*list(corr.values()))
        assert imp.shape == adatasp.shape

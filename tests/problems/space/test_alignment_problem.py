from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import pytest

import numpy as np
import pandas as pd
import scipy.sparse as sp
from ott.geometry import epsilon_scheduler

import scanpy as sc
from anndata import AnnData

from moscot.backends.ott._utils import alpha_to_fused_penalty
from moscot.problems.space import AlignmentProblem
from moscot.utils.tagged_array import Tag, TaggedArray
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_linear_solver_args,
    gw_lr_linear_solver_args,
    gw_lr_solver_args,
    gw_solver_args,
    pointcloud_args,
    quad_prob_args,
)

# TODO(giovp): refactor as fixture
SOLUTIONS_PATH = Path("./tests/data/alignment_solutions.pkl")  # base is moscot


class TestAlignmentProblem:
    @pytest.mark.fast()
    @pytest.mark.parametrize("joint_attr", [{"attr": "X"}])
    @pytest.mark.parametrize("normalize_spatial", [True, False])
    def test_prepare_sequential(
        self,
        adata_space_rotate: AnnData,
        joint_attr: Optional[Mapping[str, Any]],
        normalize_spatial: bool,
    ):
        n_obs = adata_space_rotate.shape[0] // 3  # adata is made of 3 datasets
        n_var = adata_space_rotate.shape[1]
        expected_keys = {("0", "1"), ("1", "2")}
        ap = AlignmentProblem(adata=adata_space_rotate)
        assert len(ap) == 0
        assert ap.problems == {}
        assert ap.solutions == {}

        ap = ap.prepare(batch_key="batch", joint_attr=joint_attr, normalize_spatial=normalize_spatial)
        assert len(ap) == 2
        if normalize_spatial:
            np.testing.assert_allclose(ap[("1", "2")].x.data_src.std(), ap[("0", "1")].x.data_src.std(), atol=1e-15)
            np.testing.assert_allclose(ap[("1", "2")].x.data_src.std(), 1.0, atol=1e-15)
            np.testing.assert_allclose(ap[("1", "2")].x.data_src.mean(), 0, atol=1e-15)
            np.testing.assert_allclose(ap[("0", "1")].x.data_src.mean(), 0, atol=1e-15)

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
                return  # TODO(@MUCDK) fix after refactoring
        ap = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch")
            .solve(epsilon=epsilon, alpha=alpha, rank=rank, **kwargs)
        )
        for prob_key in ap:
            assert ap[prob_key].solution.rank == rank
            if initializer != "random":  # TODO: is this valid?
                assert ap[prob_key].solution.converged

        # TODO(michalk8): use np.testing
        assert np.allclose(*(sol.cost for sol in ap.solutions.values()))
        assert np.all([sol.converged for sol in ap.solutions.values()])
        np.testing.assert_array_equal(
            [np.all(np.isfinite(sol.transport_matrix)) for sol in ap.solutions.values()], True
        )

    def test_solve_unbalanced(self, adata_space_rotate: AnnData):
        tau_a, tau_b = [0.8, 1]
        marg_a = "a"
        marg_b = "b"
        adata_space_rotate.obs[marg_a] = adata_space_rotate.obs[marg_b] = np.ones(300)
        ap: AlignmentProblem = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch", a=marg_a, b=marg_b)
            .solve(tau_a=tau_a, tau_b=tau_b)
        )
        assert np.all([sol.a is not None for sol in ap.solutions.values()])
        assert np.all([sol.b is not None for sol in ap.solutions.values()])
        assert np.all([sol.converged for sol in ap.solutions.values()])
        assert np.allclose(*(sol.cost for sol in ap.solutions.values()), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("key", ["connectivities", "distances"])
    @pytest.mark.parametrize("dense_input", [True, False])
    def test_geodesic_cost_xy(self, adata_space_rotate: AnnData, key: str, dense_input: bool):
        batch_column = "batch"
        unique_batches = adata_space_rotate.obs[batch_column].unique()

        dfs = []
        for i in range(len(unique_batches) - 1):
            batch1 = unique_batches[i]
            batch2 = unique_batches[i + 1]
            indices = np.where(
                (adata_space_rotate.obs[batch_column] == batch1) | (adata_space_rotate.obs[batch_column] == batch2)
            )[0]
            adata_subset = adata_space_rotate[indices]
            sc.pp.neighbors(adata_subset, n_neighbors=15, use_rep="X_pca")
            df = (
                pd.DataFrame(
                    index=adata_subset.obs_names,
                    columns=adata_subset.obs_names,
                    data=adata_subset.obsp[key].toarray().astype("float64"),
                )
                if dense_input
                else (
                    adata_subset.obsp[key].astype("float64"),
                    adata_subset.obs_names.to_series(),
                    adata_subset.obs_names.to_series(),
                )
            )
            dfs.append(df)

        ap: AlignmentProblem = AlignmentProblem(adata=adata_space_rotate)
        ap = ap.prepare(batch_key=batch_column, joint_attr={"attr": "obsm", "key": "X_pca"})

        ap[("0", "1")].set_graph_xy(dfs[0], cost="geodesic")
        ap[("1", "2")].set_graph_xy(dfs[1], cost="geodesic")
        ap = ap.solve(max_iterations=2, lse_mode=False)

        ta = ap[("0", "1")].xy
        assert isinstance(ta, TaggedArray)
        assert isinstance(ta.data_src, np.ndarray) if dense_input else sp.issparse(ta.data_src)
        assert ta.data_tgt is None
        assert ta.tag == Tag.GRAPH
        assert ta.cost == "geodesic"

        ta = ap[("1", "2")].xy
        assert isinstance(ta, TaggedArray)
        assert isinstance(ta.data_src, np.ndarray) if dense_input else sp.issparse(ta.data_src)
        assert ta.data_tgt is None
        assert ta.tag == Tag.GRAPH
        assert ta.cost == "geodesic"

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_space_rotate: AnnData, args_to_check: Mapping[str, Any]):
        adata_space_rotate = adata_space_rotate[adata_space_rotate.obs["batch"].isin(("0", "1"))]
        key = ("0", "1")
        problem = AlignmentProblem(adata=adata_space_rotate)
        problem = problem.prepare(batch_key="batch", joint_attr={"attr": "X"})
        problem = problem.solve(**args_to_check)

        solver = problem[key].solver.solver
        args = gw_solver_args if args_to_check["rank"] == -1 else gw_lr_solver_args
        for arg, val in args.items():
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

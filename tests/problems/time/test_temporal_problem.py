import copy
from typing import Any, List, Mapping, Optional

import pytest

import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ott.geometry import costs, epsilon_scheduler
from scipy.sparse import csr_matrix

import scanpy as sc
from anndata import AnnData

from moscot.backends.ott.output import GraphOTTOutput
from moscot.base.output import BaseSolverOutput
from moscot.base.problems import BirthDeathProblem
from moscot.problems.time import TemporalProblem
from moscot.utils.tagged_array import Tag, TaggedArray
from tests._utils import ATOL, RTOL
from tests.problems._utils import check_is_copy_multiple
from tests.problems.conftest import (
    geometry_args,
    lin_prob_args,
    lr_pointcloud_args,
    lr_sinkhorn_solver_args,
    pointcloud_args,
    sinkhorn_args_1,
    sinkhorn_args_2,
    sinkhorn_solver_args,
)


class TestTemporalProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(
            time_key="time",
            policy="sequential",
        )

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    @pytest.mark.parametrize("callback", ["local-pca", None])
    def test_solve_balanced(self, adata_time: AnnData, callback: Optional[str]):
        eps = 0.5
        joint_attr = None if callback else "X_pca"
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time)
        problem = problem.prepare("time", cost="cosine", xy_callback=callback, joint_attr=joint_attr)
        problem = problem.solve(epsilon=eps)

        assert isinstance(problem[0, 1].xy.cost, costs.Cosine)
        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_solve_unbalanced(self, adata_time: AnnData):
        taus = [9e-1, 1e-2]
        problem1 = TemporalProblem(adata=adata_time)
        problem2 = TemporalProblem(adata=adata_time)
        problem1 = problem1.prepare("time", a="left_marginals_unbalanced", b="right_marginals_unbalanced")
        problem2 = problem2.prepare("time", a="left_marginals_unbalanced", b="right_marginals_unbalanced")

        assert problem1[0, 1].a is not None
        assert problem1[0, 1].b is not None
        assert problem2[0, 1].a is not None
        assert problem2[0, 1].b is not None

        problem1 = problem1.solve(epsilon=1, tau_a=taus[0], tau_b=taus[0], max_iterations=10000)
        problem2 = problem2.solve(epsilon=1, tau_a=taus[1], tau_b=taus[1], max_iterations=10000)

        assert problem1[0, 1].solution.a is not None
        assert problem1[0, 1].solution.b is not None
        assert problem2[0, 1].solution.a is not None
        assert problem2[0, 1].solution.b is not None

        div1 = np.linalg.norm(problem1[0, 1].a - problem1[0, 1].solution.a)
        div2 = np.linalg.norm(problem1[0, 1].b - problem1[0, 1].solution.b)
        assert div1 < div2

    @pytest.mark.fast()
    @pytest.mark.parametrize(
        "gene_set_list",
        [
            [None, None],
            ["human", "human"],
            ["mouse", "mouse"],
            [["ANLN", "ANP32E", "ATAD2"], ["ADD1", "AIFM3", "ANKH"]],
        ],
    )
    def test_score_genes(self, adata_time: AnnData, gene_set_list: List[List[str]]):
        gene_set_proliferation = gene_set_list[0]
        gene_set_apoptosis = gene_set_list[1]
        problem = TemporalProblem(adata_time)
        problem.score_genes_for_marginals(
            gene_set_proliferation=gene_set_proliferation, gene_set_apoptosis=gene_set_apoptosis
        )

        if gene_set_apoptosis is not None:
            assert problem.proliferation_key == "proliferation"
            assert adata_time.obs["proliferation"] is not None
            assert np.sum(np.isnan(adata_time.obs["proliferation"])) == 0
        else:
            assert problem.proliferation_key is None

        if gene_set_apoptosis is not None:
            assert problem.apoptosis_key == "apoptosis"
            assert adata_time.obs["apoptosis"] is not None
            assert np.sum(np.isnan(adata_time.obs["apoptosis"])) == 0
        else:
            assert problem.apoptosis_key is None

    @pytest.mark.fast()
    def test_proliferation_key_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata_time)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        adata_time.obs["new_proliferation"] = np.ones(adata_time.n_obs)
        problem.proliferation_key = "new_proliferation"
        assert problem.proliferation_key == "new_proliferation"

    @pytest.mark.fast()
    def test_apoptosis_key_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata_time)
        assert problem.apoptosis_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.apoptosis_key == "apoptosis"

        adata_time.obs["new_apoptosis"] = np.ones(adata_time.n_obs)
        problem.apoptosis_key = "new_apoptosis"
        assert problem.apoptosis_key == "new_apoptosis"

    @pytest.mark.fast()
    @pytest.mark.parametrize("scaling", [0.1, 1, 4])
    def test_proliferation_key_c_pipeline(self, adata_time: AnnData, scaling: float):
        key0, key1, *_ = np.sort(np.unique(adata_time.obs["time"].values))
        adata_time = adata_time[adata_time.obs["time"].isin([key0, key1])].copy()
        delta = key1 - key0
        problem = TemporalProblem(adata_time)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        problem = problem.prepare(time_key="time", marginal_kwargs={"scaling": scaling})
        prolif = adata_time[adata_time.obs["time"] == key0].obs["proliferation"]
        apopt = adata_time[adata_time.obs["time"] == key0].obs["apoptosis"]
        expected_marginals = np.exp((prolif - apopt) * delta / scaling)
        np.testing.assert_allclose(problem[key0, key1]._prior_growth, expected_marginals, rtol=RTOL, atol=ATOL)

    def test_cell_costs_source_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata=adata_time).prepare("time")
        problem = problem.solve(max_iterations=2)

        cell_costs_source = problem.cell_costs_source

        assert isinstance(cell_costs_source, pd.DataFrame)
        assert len(cell_costs_source.columns) == 1
        assert list(cell_costs_source.columns)[0] == "cell_cost_source"
        assert set(cell_costs_source.index) == set(adata_time.obs.index)
        assert set(cell_costs_source[cell_costs_source["cell_cost_source"].isnull()].index) == set(
            adata_time[adata_time.obs["time"] == 2].obs.index
        )
        assert set(cell_costs_source[~cell_costs_source["cell_cost_source"].isnull()].index) == set(
            adata_time[adata_time.obs["time"].isin([0, 1])].obs.index
        )

    def test_cell_costs_target_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve(max_iterations=2)

        cell_costs_target = problem.cell_costs_target

        assert isinstance(cell_costs_target, pd.DataFrame)
        assert len(cell_costs_target.columns) == 1
        assert list(cell_costs_target.columns)[0] == "cell_cost_target"
        assert set(cell_costs_target.index) == set(adata_time.obs.index)
        assert set(cell_costs_target[cell_costs_target["cell_cost_target"].isnull()].index) == set(
            adata_time[adata_time.obs["time"] == 0].obs.index
        )
        assert set(cell_costs_target[~cell_costs_target["cell_cost_target"].isnull()].index) == set(
            adata_time[adata_time.obs["time"].isin([1, 2])].obs.index
        )

    def test_growth_rates_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata=adata_time)
        problem = problem.score_genes_for_marginals(gene_set_proliferation="mouse", gene_set_apoptosis="mouse")
        problem = problem.prepare("time", a=True, b=True)
        problem = problem.solve(max_iterations=2)

        growth_rates = problem.posterior_growth_rates
        assert isinstance(growth_rates, pd.DataFrame)
        assert len(growth_rates.columns) == 1
        assert set(growth_rates.index) == set(adata_time.obs.index)
        assert set(growth_rates[growth_rates["posterior_growth_rates"].isnull()].index) == set(
            adata_time[adata_time.obs["time"] == 2].obs.index
        )
        assert set(growth_rates[~growth_rates["posterior_growth_rates"].isnull()].index) == set(
            adata_time[adata_time.obs["time"].isin([0, 1])].obs.index
        )

    def test_result_compares_to_wot(self, gt_temporal_adata: AnnData):
        # this test assures TemporalProblem returns an equivalent solution to Waddington OT (precomputed)
        adata = gt_temporal_adata.copy()
        config = gt_temporal_adata.uns
        eps = config["eps"]
        lam1 = config["lam1"]
        lam2 = config["lam2"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]

        tp = TemporalProblem(adata)
        tp = tp.prepare(
            key,
            subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)],
            policy="explicit",
            xy_callback_kwargs={"n_comps": 50},
        )
        tp = tp.solve(epsilon=eps, scale_cost="mean", tau_a=lam1 / (lam1 + eps), tau_b=lam2 / (lam2 + eps))

        np.testing.assert_array_almost_equal(
            adata.uns["tmap_10_105"],
            np.array(tp[key_1, key_2].solution.transport_matrix),
        )
        np.testing.assert_array_almost_equal(
            adata.uns["tmap_105_11"],
            np.array(tp[key_2, key_3].solution.transport_matrix),
        )
        np.testing.assert_array_almost_equal(
            adata.uns["tmap_10_11"],
            np.array(tp[key_1, key_3].solution.transport_matrix),
        )

    @pytest.mark.parametrize("dense_input", [True, False])
    def test_geodesic_cost_set_xy_cost(self, adata_time, dense_input):
        # TODO(@MUCDK) add test for failure case
        tp = TemporalProblem(adata_time)
        tp = tp.prepare("time", joint_attr="X_pca")
        batch_column = "time"
        unique_batches = adata_time.obs[batch_column].unique()

        dfs = []
        for i in range(len(unique_batches) - 1):
            batch1 = unique_batches[i]
            batch2 = unique_batches[i + 1]

            indices = np.where((adata_time.obs[batch_column] == batch1) | (adata_time.obs[batch_column] == batch2))[0]
            adata_subset = adata_time[indices]
            sc.pp.neighbors(adata_subset, n_neighbors=15, use_rep="X_pca")
            df = (
                pd.DataFrame(
                    index=adata_subset.obs_names,
                    columns=adata_subset.obs_names,
                    data=adata_subset.obsp["connectivities"].A.astype("float64"),
                )
                if dense_input
                else (
                    adata_subset.obsp["connectivities"].astype("float64"),
                    adata_subset.obs_names.to_series(),
                    adata_subset.obs_names.to_series(),
                )
            )
            dfs.append(df)

        tp[0, 1].set_graph_xy(dfs[0], cost="geodesic")
        tp = tp.solve(max_iterations=2, lse_mode=False)

        ta = tp[0, 1].xy
        assert isinstance(ta, TaggedArray)
        assert isinstance(ta.data_src, np.ndarray) if dense_input else sp.issparse(ta.data_src)
        assert ta.data_tgt is None
        assert ta.tag == Tag.GRAPH
        assert ta.cost == "geodesic"

        tp[1, 2].set_graph_xy(dfs[1], cost="geodesic")
        tp = tp.solve(max_iterations=2, lse_mode=False)

        ta = tp[1, 2].xy
        assert isinstance(ta, TaggedArray)
        assert isinstance(ta.data_src, np.ndarray) if dense_input else sp.issparse(ta.data_src)
        assert ta.data_tgt is None
        assert ta.tag == Tag.GRAPH
        assert ta.cost == "geodesic"

    def test_geodesic_cost_set_xy_cost_sparse(self, adata_time):
        tp = TemporalProblem(adata_time)
        tp = tp.prepare("time", joint_attr="X_pca")
        batch_column = "time"
        unique_batches = adata_time.obs[batch_column].unique()

        elements = []
        for i in range(len(unique_batches) - 1):
            batch1 = unique_batches[i]
            batch2 = unique_batches[i + 1]

            indices = np.where((adata_time.obs[batch_column] == batch1) | (adata_time.obs[batch_column] == batch2))[0]
            adata_subset = adata_time[indices]
            sc.pp.neighbors(adata_subset, n_neighbors=15, use_rep="X_pca")

            sparse_matrix = adata_subset.obsp["connectivities"].astype("float64")
            row_names = adata_subset.obs_names.to_series()
            col_names = adata_subset.obs_names.to_series()
            elements.append((sparse_matrix, row_names, col_names))

        tp[0, 1].set_graph_xy(elements[0], cost="geodesic")
        tp = tp.solve(max_iterations=2, lse_mode=False)

        ta = tp[0, 1].xy
        assert isinstance(ta, TaggedArray)
        assert isinstance(ta.data_src, csr_matrix)
        assert ta.data_tgt is None
        assert ta.tag == Tag.GRAPH
        assert ta.cost == "geodesic"

        tp[1, 2].set_graph_xy(elements[1], cost="geodesic")
        tp = tp.solve(max_iterations=2, lse_mode=False)

        ta = tp[1, 2].xy
        assert isinstance(ta, TaggedArray)
        assert isinstance(ta.data_src, csr_matrix)
        assert ta.data_tgt is None
        assert ta.tag == Tag.GRAPH
        assert ta.cost == "geodesic"

    @pytest.mark.parametrize("callback_kwargs", [{}, {"n_neighbors": 3}, {"foo": "bar"}])
    def test_graph_construction_callback(self, adata_time: AnnData, callback_kwargs: Mapping[str, Any]):
        eps = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time)

        if "foo" in callback_kwargs:
            with pytest.raises(TypeError):
                problem = problem.prepare(
                    "time", cost="geodesic", xy_callback="graph-construction", xy_callback_kwargs=callback_kwargs
                )
            return
        problem = problem.prepare(
            "time", cost="geodesic", xy_callback="graph-construction", xy_callback_kwargs=callback_kwargs
        )

        problem = problem.solve(epsilon=eps, lse_mode=False)

        assert problem[0, 1].xy.cost == "geodesic"
        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

        if "n_neighbors" in callback_kwargs:
            callback_kwargs["n_neighbors"] = callback_kwargs["n_neighbors"] + 20
            problem2 = TemporalProblem(adata=adata_time)
            problem2 = problem2.prepare(
                "time", cost="geodesic", xy_callback="graph-construction", xy_callback_kwargs=callback_kwargs
            )

            assert np.sum(problem2[0, 1].xy.data_src.sum(axis=1) != problem[0, 1].xy.data_src.sum(axis=1)) > 0
            assert np.all(problem2[0, 1].xy.data_src.sum(axis=1) > problem[0, 1].xy.data_src.sum(axis=1))

    @pytest.mark.parametrize("forward", [True, False])
    def test_geodesic_cost_downstream(self, adata_time: AnnData, forward: bool):
        # TODO(@MUCDK) add test for failure case
        adata_time = adata_time[adata_time.obs["time"].isin([0, 1])]
        tp = TemporalProblem(adata_time)
        tp = tp.prepare("time", joint_attr="X_pca")
        batch_column = "time"
        unique_batches = adata_time.obs[batch_column].unique()

        dfs = []
        for i in range(len(unique_batches) - 1):
            batch1 = unique_batches[i]
            batch2 = unique_batches[i + 1]

            indices = np.where((adata_time.obs[batch_column] == batch1) | (adata_time.obs[batch_column] == batch2))[0]
            adata_subset = adata_time[indices]
            sc.pp.neighbors(adata_subset, n_neighbors=len(adata_subset), use_rep="X_pca")
            df = pd.DataFrame(
                index=adata_subset.obs_names,
                columns=adata_subset.obs_names,
                data=adata_subset.obsp["connectivities"].A.astype("float64"),
            )
            order = pd.concat(
                (tp[batch1, batch2].adata_src.obs_names.to_series(), tp[batch1, batch2].adata_tgt.obs_names.to_series())
            )
            df = df.loc[order, :]
            df = df.loc[:, order]
            dfs.append(df)

        tp[0, 1].set_graph_xy(dfs[0], cost="geodesic")
        tp = tp.solve(max_iterations=5, lse_mode=False)
        assert isinstance(tp[0, 1].solution, GraphOTTOutput)

        ta = tp[0, 1].xy
        assert isinstance(ta, TaggedArray)
        assert isinstance(ta.data_src, np.ndarray)
        assert ta.data_tgt is None
        assert ta.tag == Tag.GRAPH
        assert ta.cost == "geodesic"

        func = tp.push if forward else tp.pull
        out = func(0, 1, "celltype", "A", key_added=None)
        assert isinstance(out, jnp.ndarray)
        assert jnp.sum(jnp.isnan(out)) == 0

        adata_time.obs["celltype"] = adata_time.obs["celltype"].astype("category")
        df = tp.cell_transition(0, 1, "celltype", "celltype", forward=forward)
        assert isinstance(df, pd.DataFrame)
        assert df.isna().sum().sum() == 0
        assert df.sum().sum() > 0

    @pytest.mark.parametrize("args_to_check", [sinkhorn_args_1, sinkhorn_args_2])
    def test_pass_arguments(self, adata_time: AnnData, args_to_check: Mapping[str, Any]):
        problem = TemporalProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]

        problem = problem.prepare(
            time_key="time",
            policy="sequential",
        )

        problem = problem.solve(**args_to_check)
        key = (0, 1)
        solver = problem[key].solver.solver
        args = sinkhorn_solver_args if args_to_check["rank"] == -1 else lr_sinkhorn_solver_args
        for arg, val in args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == args_to_check[arg]

        lin_prob = problem[key]._solver._problem
        for arg, val in lin_prob_args.items():
            assert hasattr(lin_prob, val)
            el = getattr(lin_prob, val)[0] if isinstance(getattr(lin_prob, val), tuple) else getattr(lin_prob, val)
            assert el == args_to_check[arg]

        geom = lin_prob.geom
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

        args = pointcloud_args if args_to_check["rank"] == -1 else lr_pointcloud_args
        for arg, val in args.items():
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            assert hasattr(geom, val)
            if arg == "cost":
                assert type(el) == type(args_to_check[arg])  # noqa: E721
            else:
                assert el == args_to_check[arg]

    def test_copy(self, adata_time: AnnData):
        shallow_copy = ("_adata",)

        eps = 0.5
        prepare_params = {"time_key": "time", "cost": "cosine", "xy_callback": None, "joint_attr": "X_pca"}
        solve_params = {"epsilon": eps}

        prob = TemporalProblem(adata=adata_time)
        prob_copy_1 = prob.copy()

        assert check_is_copy_multiple((prob, prob_copy_1), shallow_copy)

        prob = prob.prepare(**prepare_params)  # type: ignore
        prob_copy_1 = prob_copy_1.prepare(**prepare_params)  # type: ignore
        prob_copy_2 = prob.copy()

        assert check_is_copy_multiple((prob, prob_copy_1, prob_copy_2), shallow_copy)

        prob = prob.solve(**solve_params)  # type: ignore
        with pytest.raises(copy.Error):
            _ = prob.copy()

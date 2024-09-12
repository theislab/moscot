from contextlib import nullcontext
from typing import Any, Literal, Mapping, Optional, Tuple

import pytest

import numpy as np
from ott.geometry import epsilon_scheduler

from anndata import AnnData

from moscot.backends.ott._utils import alpha_to_fused_penalty
from moscot.base.output import BaseDiscreteSolverOutput
from moscot.problems.cross_modality import TranslationProblem
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


class TestTranslationProblem:
    @pytest.mark.fast
    @pytest.mark.parametrize("src_attr", ["emb_src", {"attr": "obsm", "key": "emb_src"}])
    @pytest.mark.parametrize("tgt_attr", ["emb_tgt", {"attr": "obsm", "key": "emb_tgt"}])
    @pytest.mark.parametrize("joint_attr", [None, "X_pca", {"attr": "obsm", "key": "X_pca"}])
    def test_prepare_dummy_policy(
        self,
        adata_translation_split: Tuple[AnnData, AnnData],
        src_attr: Mapping[str, str],
        tgt_attr: Mapping[str, str],
        joint_attr: Optional[Mapping[str, str]],
    ):
        adata_src, adata_tgt = adata_translation_split
        n_obs = adata_tgt.shape[0]

        tp = TranslationProblem(adata_src, adata_tgt)
        assert tp.problems == {}
        assert tp.solutions == {}

        prob_key = ("src", "tgt")
        tp = tp.prepare(src_attr=src_attr, tgt_attr=tgt_attr, joint_attr=joint_attr)
        assert len(tp) == 1
        assert isinstance(tp[prob_key], tp._base_problem_type)
        assert tp[prob_key].shape == (2 * n_obs, n_obs)
        np.testing.assert_array_equal(tp._policy._cat, prob_key)

    @pytest.mark.fast
    @pytest.mark.parametrize("src_attr", ["emb_src", {"attr": "obsm", "key": "emb_src"}])
    @pytest.mark.parametrize("tgt_attr", ["emb_tgt", {"attr": "obsm", "key": "emb_tgt"}])
    @pytest.mark.parametrize("joint_attr", [None, "X_pca", {"attr": "obsm", "key": "X_pca"}])
    def test_prepare_external_star_policy(
        self,
        adata_translation_split: Tuple[AnnData, AnnData],
        src_attr: Mapping[str, str],
        tgt_attr: Mapping[str, str],
        joint_attr: Optional[Mapping[str, str]],
    ):
        adata_src, adata_tgt = adata_translation_split
        expected_keys = {(i, "ref") for i in adata_src.obs["batch"]}
        n_obs = adata_tgt.shape[0]
        x_n_var = adata_src.obsm["emb_src"].shape[1]
        y_n_var = adata_tgt.obsm["emb_tgt"].shape[1]
        xy_n_vars = adata_src.X.shape[1] if joint_attr == "default" else adata_src.obsm["X_pca"].shape[1]
        tp = TranslationProblem(adata_src, adata_tgt)
        assert tp.problems == {}
        assert tp.solutions == {}

        tp = tp.prepare(batch_key="batch", src_attr=src_attr, tgt_attr=tgt_attr, joint_attr=joint_attr)

        assert len(tp) == len(expected_keys)
        for prob_key in expected_keys:
            assert isinstance(tp[prob_key], tp._base_problem_type)
            assert tp[prob_key].shape == (n_obs, n_obs)
            assert tp[prob_key].x.data_src.shape == (n_obs, x_n_var)
            assert tp[prob_key].y.data_src.shape == (n_obs, y_n_var)
            if joint_attr is not None:
                assert tp[prob_key].xy.data_src.shape == tp[prob_key].xy.data_tgt.shape == (n_obs, xy_n_vars)

    @pytest.mark.parametrize(
        ("epsilon", "alpha", "rank", "initializer", "joint_attr", "expect_fail"),
        [
            (1e-2, 0.9, -1, None, {"attr": "obsm", "key": "X_pca"}, False),
            (2, 0.5, -1, "random", None, True),
            (2, 0.5, -1, "random", {"attr": "X"}, False),
            (2, 1.0, -1, "rank2", None, False),
            (2, 1.0, -1, "rank2", {"attr": "obsm", "key": "X_pca"}, True),
            (2, 0.1, -1, None, {"attr": "obsm", "key": "X_pca"}, False),
            (2, 1.0, -1, None, None, False),
            (1.3, 1.0, -1, "random", None, False),
        ],
    )
    @pytest.mark.parametrize("src_attr", ["emb_src", {"attr": "obsm", "key": "emb_src"}])
    @pytest.mark.parametrize("tgt_attr", ["emb_tgt", {"attr": "obsm", "key": "emb_tgt"}])
    def test_solve_balanced(
        self,
        adata_translation_split: Tuple[AnnData, AnnData],
        epsilon: float,
        alpha: float,
        rank: int,
        src_attr: Mapping[str, str],
        tgt_attr: Mapping[str, str],
        initializer: Optional[Literal["random", "rank2"]],
        joint_attr: Optional[Mapping[str, str]],
        expect_fail: bool,
    ):
        adata_src, adata_tgt = adata_translation_split
        kwargs = {}
        expected_keys = {(i, "ref") for i in adata_src.obs["batch"]}
        if rank > -1:
            kwargs["initializer"] = initializer

        tp = TranslationProblem(adata_src, adata_tgt)
        tp = tp.prepare(batch_key="batch", src_attr=src_attr, tgt_attr=tgt_attr, joint_attr=joint_attr)
        context = pytest.raises(ValueError, match="alpha") if expect_fail else nullcontext()
        with context:
            tp = tp.solve(epsilon=epsilon, alpha=alpha, rank=rank, **kwargs)

        for key, subsol in tp.solutions.items():
            assert isinstance(subsol, BaseDiscreteSolverOutput)
            assert key in expected_keys
            assert tp[key].solution.rank == rank

        for key, sol in tp.solutions.items():
            np.testing.assert_array_equal(np.isfinite(sol.transport_matrix), True, err_msg=f"{key}")

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_translation_split: Tuple[AnnData, AnnData], args_to_check: Mapping[str, Any]):
        adata_src, adata_tgt = adata_translation_split
        tp = TranslationProblem(adata_src, adata_tgt)

        key = ("1", "ref")
        tp = tp.prepare(
            batch_key="batch",
            src_attr={"attr": "obsm", "key": "emb_src"},
            tgt_attr={"attr": "obsm", "key": "emb_tgt"},
            joint_attr="X_pca",
        )
        tp = tp.solve(**args_to_check)

        solver = tp[key].solver.solver

        args = gw_solver_args if args_to_check["rank"] == -1 else gw_lr_solver_args
        for arg, val in args.items():
            assert getattr(solver, val) == args_to_check[arg], arg

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

        quad_prob = tp[key]._solver._problem
        for arg, val in quad_prob_args.items():
            assert getattr(quad_prob, val) == args_to_check[arg], arg
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
            assert getattr(geom, val) == args_to_check[arg], arg

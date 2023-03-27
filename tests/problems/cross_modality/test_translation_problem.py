from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional

import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.cross_modality import TranslationProblem
from tests._utils import _adata_modality_split
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

class TestTranslationProblem:
    @pytest.mark.fast()
    @pytest.mark.parametrize("src_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize("tgt_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize("joint_attr", [None, "X_pca", {"attr": "obsm", "key": "X_pca"}])
    def test_prepare(
        self, 
        adata_space_rotate: AnnData, 
        src_attr: Mapping[str, str], 
        tgt_attr: Mapping[str, str], 
        joint_attr: Optional[Mapping[str, str]]
        ):

        adata_src, adata_tgt = _adata_modality_split(adata_space_rotate)
        expected_keys = ("src", "tgt") # {(i, "ref") for i in adata_src.obs.batch.cat.categories}
        n_obs = adata_tgt.shape[0]
        x_n_var = adata_src.obsm["spatial"].shape[1]
        y_n_var = adata_tgt.shape[1] if sc_attr["attr"] == "X" else adata_tgt.obsm["X_pca"].shape[1]
        xy_n_vars = adata_src.X.shape[1] if joint_attr == "default" else adata_tgt.obsm["X_pca"].shape[1]
        
        tp = TranslationProblem(adata_src, adata_tgt)
        assert tp.problems == {}
        assert tp.solutions == {}

        tp = tp.prepare(src_attr = src_attr, tgt_attr = tgt_attr, joint_attr=joint_attr)

        assert len(tp) == len(expected_keys)
        for prob_key in expected_keys:
            assert isinstance(tp[prob_key], tp._base_problem_type)
            assert tp[prob_key].shape == (n_obs, n_obs)
            assert tp[prob_key].x.data_src.shape == (n_obs, x_n_var)
            assert tp[prob_key].y.data_src.shape == (n_obs, y_n_var)
            assert tp[prob_key].xy.data_src.shape == tp[prob_key].xy.data_tgt.shape == (n_obs, xy_n_vars)

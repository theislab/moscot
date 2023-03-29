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
        n_obs_src = adata_src.shape[0]        
        n_obs_tgt = adata_tgt.shape[0]
        
        tp = TranslationProblem(adata_src, adata_tgt)
        assert tp.problems == {}
        assert tp.solutions == {}

        # test dummy policy
        prob_key = ("src", "tgt")
        tp = tp.prepare(src_attr = src_attr, tgt_attr = tgt_attr, joint_attr=joint_attr)

        assert len(tp) == 1
        assert isinstance(tp[prob_key], tp._base_problem_type)
        assert tp[prob_key].shape == (n_obs_src, n_obs_tgt)
        np.testing.assert_array_equal(tp._policy._cat, prob_key)

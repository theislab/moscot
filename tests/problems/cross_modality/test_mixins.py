import pickle
from math import acos
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional

import pytest

import numpy as np
import pandas as pd

from anndata import AnnData

from moscot.problems.cross_modality import TranslationProblem
from tests._utils import MockSolverOutput, _adata_modality_split
from tests.conftest import ANGLES

class TestCrossModalityTranslationAnalysisMixin:
    @pytest.mark.parametrize("src_attr", ["emb_src", {"attr": "obsm", "key": "emb_src"}])
    @pytest.mark.parametrize("tgt_attr", ["emb_tgt", {"attr": "obsm", "key": "emb_tgt"}])
    @pytest.mark.parametrize("joint_attr", [None, "X_pca", {"attr": "obsm", "key": "X_pca"}])
    def test_translate(
        self,
        adata_translation: AnnData,
        src_attr: Mapping[str, str], 
        tgt_attr: Mapping[str, str], 
        joint_attr: Optional[Mapping[str, str]]
    ):
        adata_tgt, adata_src = _adata_modality_split(adata_translation)
        expected_keys = {(i, "ref") for i in adata_src.obs.batch.cat.categories}
        
        tp = TranslationProblem(adata_src, adata_tgt).prepare(batch_key="batch", src_attr = src_attr, tgt_attr = tgt_attr, joint_attr=joint_attr).solve()
        for prob_key in expected_keys:
            trans_forward = tp.translate(source=prob_key[0], target=prob_key[1], forward=True)
            assert trans_forward.shape == tp[prob_key].y.data_src.shape
            trans_backward = tp.translate(source=prob_key[0], target=prob_key[1], forward=False)
            assert trans_backward.shape == tp[prob_key].x.data_src.shape

    @pytest.mark.fast()
    @pytest.mark.parametrize("forward", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_cell_transition_pipeline(self, adata_translation: AnnData, forward: bool, normalize: bool):
        rng = np.random.RandomState(0)
        adata_tgt, adata_src = _adata_modality_split(adata_translation)
        tp = TranslationProblem(adata_src, adata_tgt)
        tp = tp.prepare(batch_key="batch", src_attr = 'emb_src', tgt_attr = 'emb_tgt', joint_attr='X_pca')
        mock_tmap_1 = np.abs(rng.randn(len(adata_src[adata_src.obs["batch"] == "1"]), len(adata_tgt)))
        mock_tmap_2 = np.abs(rng.randn(len(adata_src[adata_src.obs["batch"] == "2"]), len(adata_tgt)))
        tp[("1", "ref")]._solution = MockSolverOutput(mock_tmap_1 / np.sum(mock_tmap_1))
        tp[("2", "ref")]._solution = MockSolverOutput(mock_tmap_2 / np.sum(mock_tmap_2))

        result1 = tp.cell_transition(
            source="1",
            source_groups="celltype",
            target_groups="celltype",
            forward=forward,
            normalize=normalize,
        )

        result2 = tp.cell_transition(
            source="2",
            source_groups="celltype",
            target_groups="celltype",
            forward=forward,
            normalize=normalize,
        )

        assert isinstance(result1, pd.DataFrame)
        assert result1.shape == (3, 3)
        assert isinstance(result2, pd.DataFrame)
        assert result2.shape == (3, 3)
        assert result1.all != result2.all
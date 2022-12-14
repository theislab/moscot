from typing import List, Optional
import os

import pytest

import numpy as np

from anndata import AnnData

from tests._utils import Problem
from moscot.plotting._utils import _input_to_adatas
import moscot.plotting as mpl


class TestMoscotPl:
    def test_input_to_adatas_problem(self, adata_time: AnnData):
        p = Problem(adata_time)
        adata1, adata2 = _input_to_adatas(p)
        assert isinstance(adata1, AnnData)
        assert isinstance(adata2, AnnData)
        np.testing.assert_array_equal(adata1.X.A, adata_time.X.A)
        np.testing.assert_array_equal(adata2.X.A, adata_time.X.A)

    def test_input_to_adatas_adata(self, adata_time: AnnData):
        adata1, adata2 = _input_to_adatas(adata_time)
        assert isinstance(adata1, AnnData)
        assert isinstance(adata2, AnnData)
        np.testing.assert_array_equal(adata1.X.A, adata_time.X.A)
        np.testing.assert_array_equal(adata2.X.A, adata_time.X.A)

    @pytest.mark.parametrize("save", [None, "tests/data/test_plot.png"])
    @pytest.mark.parametrize("return_fig", [True, False])
    def test_cell_transition(self, adata_pl_cell_transition: AnnData, return_fig: bool, save: Optional[str]):
        if save:
            if os.path.exists(save):
                os.remove(save)
        fig = mpl.cell_transition(adata_pl_cell_transition, return_fig=return_fig, save=save)
        if return_fig:
            assert fig is not None
        else:
            assert fig is None
        if save:
            assert os.path.exists(save)

    @pytest.mark.parametrize("time_points", [None, [0]])
    @pytest.mark.parametrize("return_fig", [True, False])
    @pytest.mark.parametrize("save", [None, "tests/data/test_plot.png"])
    def test_push(
        self, adata_pl_push: AnnData, time_points: Optional[List[int]], return_fig: bool, save: Optional[str]
    ):
        if save:
            if os.path.exists(save):
                os.remove(save)

        fig = mpl.push(adata_pl_push, time_points=time_points, return_fig=return_fig, save=save)

        if return_fig:
            assert fig is not None
        else:
            assert fig is None
        if save:
            assert os.path.exists(save)

    @pytest.mark.parametrize("time_points", [None, [0]])
    @pytest.mark.parametrize("return_fig", [True, False])
    @pytest.mark.parametrize("save", [None, "tests/data/test_plot.png"])
    def test_pull(
        self, adata_pl_pull: AnnData, time_points: Optional[List[int]], return_fig: bool, save: Optional[str]
    ):
        if save:
            if os.path.exists(save):
                os.remove(save)
        fig = mpl.pull(adata_pl_pull, time_points=time_points, return_fig=return_fig, save=save)
        if return_fig:
            assert fig is not None
        else:
            assert fig is None
        if save:
            assert os.path.exists(save)

    @pytest.mark.parametrize("save", [None, "tests/data/test_plot.png"])
    @pytest.mark.parametrize("return_fig", [True, False])
    def test_sankey(self, adata_pl_sankey: AnnData, return_fig: bool, save: Optional[str]):
        if save:
            if os.path.exists(save):
                os.remove(save)
        fig = mpl.sankey(adata_pl_sankey, return_fig=return_fig, save=save)
        if return_fig:
            assert fig is not None
        else:
            assert fig is None
        if save:
            assert os.path.exists(save)

from typing import List, Optional

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

    def test_cell_transition(self, adata_pl_cell_transition: AnnData):
        mpl.cell_transition(adata_pl_cell_transition)
        mpl.cell_transition(adata_pl_cell_transition)

    @pytest.mark.parametrize("time_points", [None, [0]])
    def test_push(self, adata_pl_push: AnnData, time_points: Optional[List[int]]):
        _ = mpl.push(adata_pl_push, time_points=time_points)

    @pytest.mark.parametrize("time_points", [None, [0]])
    def test_pull(self, adata_pl_pull: AnnData, time_points: Optional[List[int]]):
        _ = mpl.pull(adata_pl_pull, time_points=time_points)

    def test_sankey(self, adata_pl_sankey: AnnData):
        mpl.sankey(adata_pl_sankey)

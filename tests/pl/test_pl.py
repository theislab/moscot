from time import time
from typing import List, Mapping, Optional
from pathlib import Path

import pytest

import numpy as np

from anndata import AnnData

from tests._utils import _adata_spatial_split, Problem
from moscot.problems.space import MappingProblem
from moscot.solvers._base_solver import ProblemKind
from moscot.pl._utils import _input_to_adatas
import moscot.pl as mpl


class TestMoscotPl:

    def test_input_to_adatas_problem(adata_time: AnnData):
        p = Problem(adata_time)
        adata = _input_to_adatas(p)
        assert isinstance(adata, AnnData)

    def test_input_to_adatas_adata(adata_time: AnnData):
        adata = _input_to_adatas(adata_time)
        assert isinstance(adata, AnnData)
        assert adata == adata_time

    def test_cell_transition(adata_pl_cell_transition: AnnData):
        plot_1 = mpl.cell_transition(adata_pl_cell_transition)
        plot_2 = mpl.cell_transition(adata_pl_cell_transition, key_stored="cell_transition_backward")

    @pytest.mark.parametrize("time_points", [None, 0])
    def test_push(adata_pl_push: AnnData, time_points: Optional[List[int]]):
        plot = mpl.push(adata_pl_push, time_points=time_points)

    @pytest.mark.parametrize("time_points", [None, 0])
    def test_pull(adata_pl_pull: AnnData, time_points: Optional[List[int]]):
        plot = mpl.pull(adata_pl_pull, time_points=time_points)
    
    def test_sankey(adata_pl_sankey: AnnData):
        plot = mpl.sankey(adata_pl_sankey)


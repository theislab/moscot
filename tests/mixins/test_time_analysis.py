from configparser import Interpolation
from multiprocessing.sharedctypes import Value
from typing import List, Tuple, Optional

import pytest

import numpy as np
import pandas as pd
from anndata import AnnData
from numbers import Number

from moscot.backends.ott import SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._lineage import TemporalProblem, TemporalBaseProblem
from _utils import TestSolverOutput
from conftest import ATOL, RTOL, Geom_t

@pytest.mark.parametrize("forward", [True, False])
def test_cell_transition_pipeline(adata_time_cell_type: AnnData, random_transport_matrix: np.ndarray, forward: bool):
    problem = TemporalProblem(adata_time_cell_type)
    problem.prepare("time", subset=[0,1])
    problem._solution = TestSolverOutput(random_transport_matrix)

    result = problem.cell_transition(0, 1, forward)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3,3)
    assert list(result.index) == ["cell_A", "cell_B", "cell_C"]
    assert list(result.columns) == ["cell_A", "cell_B", "cell_C"]
    assert np.sum(np.isnan(result)) == 0
    
    np.testing.assert_almost_equal(result.sum(axis=1 if forward else 0), 1, decimal=7)

@pytest.mark.parametrize("only_start", [True, False])
def test_get_data(adata_time: AnnData, only_start: bool):
    problem = TemporalProblem(adata_time)
    problem.prepare("time")

    result = problem._get_data(0, only_start=only_start) if only_start else problem._get_data(0, 1, 2)

    assert isinstance(result, Tuple)
    assert len(result) == 2 if only_start else len(result) == 5
    if only_start:
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], AnnData)
    else:
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)
        assert isinstance(result[3], AnnData)
        assert isinstance(result[4], np.ndarray)


@pytest.mark.parametrize("time_points", [(0,1,2), (0,2,1), ()])
def tet_get_interp_param(adata_time: AnnData, time_points: Tuple[Number]):
    start, intermediate, end = time_points if len(time_points) else (42, 43, 44)
    interpolation_parameter = None if len(time_points) == 3 else 0.5
    problem = TemporalProblem(adata_time)
    problem.prepare("time")

    if intermediate >= start or end >= intermediate:
        with np.testing.assert_raises(ValueError):
            problem._get_interp_param(interpolation_parameter, start, intermediate, end)
    else:
        inter_param = problem._get_interp_param(interpolation_parameter, start, intermediate, end)

    assert inter_param == 0.5






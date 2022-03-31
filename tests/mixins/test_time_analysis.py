from typing import Tuple
from numbers import Number

import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.time._lineage import TemporalProblem


@pytest.mark.parametrize("forward", [True, False])
def test_cell_transition_pipeline(adata_time_cell_type: AnnData, forward: bool):
    adata_time_cell_type.obs["cell_type"] = adata_time_cell_type.obs["cell_type"].astype("category")
    cell_types = set(np.unique(adata_time_cell_type.obs["cell_type"]))
    problem = TemporalProblem(adata_time_cell_type)
    problem = problem.prepare("time", subset=[0, 1])
    problem = problem.solve()

    result = problem.cell_transition(0, 1, "cell_type", "cell_type", forward=forward)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert set(result.index) == cell_types
    assert set(result.columns) == cell_types
    assert result.isna().sum().sum() == 0

    np.testing.assert_almost_equal(result.sum(axis=1 if forward else 0).values, np.ones(len(cell_types)), decimal=5)


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


@pytest.mark.parametrize("time_points", [(0, 1, 2), (0, 2, 1), ()])
def tet_get_interp_param(adata_time: AnnData, time_points: Tuple[Number]):
    start, intermediate, end = time_points if len(time_points) else (42, 43, 44)
    interpolation_parameter = None if len(time_points) == 3 else 0.5
    problem = TemporalProblem(adata_time)
    problem.prepare("time")
    problem.solve()

    if intermediate >= start or end >= intermediate:
        with np.testing.assert_raises(ValueError):
            problem._get_interp_param(interpolation_parameter, start, intermediate, end)
    else:
        inter_param = problem._get_interp_param(interpolation_parameter, start, intermediate, end)

    assert inter_param == 0.5

from math import acos
from typing import Dict, List, Optional
from pathlib import Path
import pickle

import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from tests.conftest import ANGLES, _adata_spatial_split
from moscot.problems.space import MappingProblem, AlignmentProblem

# TODO(giovp): refactor as fixture
SOLUTIONS_PATH_ALIGNMENT = Path(__file__).parent.parent.parent / "data/alignment_solutions.pkl"  # base is moscot
SOLUTIONS_PATH_MAPPING = Path(__file__).parent.parent.parent / "data/mapping_solutions.pkl"


class TestSpatialAlignmentAnalysisMixin:
    def test_analysis(self, adata_space_rotate: AnnData):
        adata_ref = adata_space_rotate.copy()
        problem = AlignmentProblem(adata=adata_ref).prepare(batch_key="batch").solve(epsilon=1e-2)
        categories = adata_space_rotate.obs.batch.cat.categories

        for ref in categories:
            problem.align(reference=ref, mode="affine")
            problem.align(reference=ref, mode="warp")
            for c1, c2 in zip(categories, categories[1:]):
                np.testing.assert_array_almost_equal(
                    adata_ref[adata_ref.obs.batch == c1].obsm["spatial_warp"],
                    adata_ref[adata_ref.obs.batch == c2].obsm["spatial_warp"],
                    decimal=6,
                )
                np.testing.assert_array_almost_equal(
                    adata_ref[adata_ref.obs.batch == c1].obsm["spatial_affine"],
                    adata_ref[adata_ref.obs.batch == c2].obsm["spatial_affine"],
                    decimal=6,
                )
                angles = sorted(
                    round(np.rad2deg(acos(arr[0, 0])))
                    for arr in adata_ref.uns["spatial"]["alignment_metadata"].values()
                    if isinstance(arr, np.ndarray)
                )
                assert set(angles).issubset(ANGLES)
            assert adata_ref.obsm["spatial_warp"].shape == adata_space_rotate.obsm["spatial"].shape

    def test_regression_testing(self, adata_space_rotate: AnnData):
        ap = AlignmentProblem(adata=adata_space_rotate).prepare(batch_key="batch").solve(alpha=0.5, epsilon=1)
        # TODO(giovp): unnecessary assert
        assert SOLUTIONS_PATH_ALIGNMENT.exists()
        with open(SOLUTIONS_PATH_ALIGNMENT, "rb") as fname:
            sol = pickle.load(fname)

        assert sol.keys() == ap.solutions.keys()
        for k in sol:
            np.testing.assert_almost_equal(sol[k].cost, ap.solutions[k].cost, decimal=1)
            np.testing.assert_almost_equal(sol[k].transport_matrix, ap.solutions[k].transport_matrix, decimal=3)


class TestSpatialMappingAnalysisMixin:
    @pytest.mark.parametrize("sc_attr", [{"attr": "X"}, {"attr": "obsm", "key": "X_pca"}])
    @pytest.mark.parametrize("var_names", ["0", [str(i) for i in range(20)]])
    def test_analysis(
        self,
        adata_mapping: AnnData,
        sc_attr: Dict[str, str],
        var_names: Optional[List[Optional[str]]],
    ):
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        mp = MappingProblem(adataref, adatasp).prepare(batch_key="batch", sc_attr=sc_attr).solve()

        corr = mp.correlate(var_names)
        imp = mp.impute()
        pd.testing.assert_series_equal(*list(corr.values()))
        assert imp.shape == adatasp.shape

    def test_regression_testing(self, adata_mapping: AnnData):
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        mp = MappingProblem(adataref, adatasp)
        mp = mp.prepare(batch_key="batch", sc_attr={"attr": "X"})
        mp = mp.solve()
        assert SOLUTIONS_PATH_MAPPING.exists()
        with open(SOLUTIONS_PATH_MAPPING, "rb") as fname:
            sol = pickle.load(fname)

        assert sol.keys() == mp.solutions.keys()
        for k in sol:
            np.testing.assert_almost_equal(sol[k].cost, mp.solutions[k].cost, decimal=1)
            np.testing.assert_almost_equal(sol[k].transport_matrix, mp.solutions[k].transport_matrix, decimal=3)

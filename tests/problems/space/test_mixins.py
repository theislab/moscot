import pickle
from math import acos
from pathlib import Path
from typing import Dict, List, Optional

import pytest

import numpy as np
import pandas as pd

from anndata import AnnData

from moscot.problems.space import AlignmentProblem, MappingProblem
from tests._utils import MockSolverOutput, _adata_spatial_split
from tests.conftest import ANGLES

# TODO(giovp): refactor as fixture
SOLUTIONS_PATH_ALIGNMENT = Path(__file__).parent.parent.parent / "data/alignment_solutions.pkl"  # base is moscot
SOLUTIONS_PATH_MAPPING = Path(__file__).parent.parent.parent / "data/mapping_solutions.pkl"


class TestSpatialAlignmentAnalysisMixin:
    def test_analysis(self, adata_space_rotate: AnnData):
        import scanpy as sc

        adata_ref = adata_space_rotate.copy()
        sc.pp.subsample(adata_ref, fraction=0.9)
        problem = AlignmentProblem(adata=adata_ref).prepare(batch_key="batch").solve(epsilon=1e-1)
        categories = adata_space_rotate.obs.batch.cat.categories

        for ref in categories:
            problem.align(reference=ref, mode="affine", key_added="spatial_affine")
            problem.align(reference=ref, mode="warp", key_added="spatial_warp")
            tgts = set(categories) - set(ref)
            for c in zip(tgts):
                assert (
                    adata_ref[adata_ref.obs.batch == c].obsm["spatial_warp"].shape
                    == adata_ref[adata_ref.obs.batch == c].obsm["spatial_affine"].shape
                )
            angles = sorted(
                round(np.rad2deg(acos(arr[0, 0])), 3)
                for arr in adata_ref.uns["spatial_affine"]["alignment_metadata"].values()
                if isinstance(arr, np.ndarray)
            )
            assert np.sum(angles) <= np.sum(ANGLES) + 2

            problem.align(reference=ref, mode="affine", spatial_key="spatial")
            for c in zip(tgts):
                assert (
                    adata_ref[adata_ref.obs.batch == c].obsm["spatial_affine"].shape
                    == adata_ref[adata_ref.obs.batch == c].obsm["spatial"].shape
                )

    def test_regression_testing(self, adata_space_rotate: AnnData):
        ap = AlignmentProblem(adata=adata_space_rotate).prepare(batch_key="batch").solve(alpha=0.5, epsilon=1)
        # TODO(giovp): unnecessary assert
        assert SOLUTIONS_PATH_ALIGNMENT.exists()
        with open(SOLUTIONS_PATH_ALIGNMENT, "rb") as fname:
            sol = pickle.load(fname)

        assert sol.keys() == ap.solutions.keys()
        for k in sol:
            np.testing.assert_almost_equal(sol[k].cost, ap.solutions[k].cost, decimal=1)
            np.testing.assert_almost_equal(
                np.array(sol[k].transport_matrix), np.array(ap.solutions[k].transport_matrix), decimal=3
            )

    @pytest.mark.fast()
    @pytest.mark.parametrize("forward", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_cell_transition_pipeline(self, adata_space_rotate: AnnData, forward: bool, normalize: bool):
        rng = np.random.RandomState(0)
        adata_space_rotate.obs["celltype"] = rng.choice(["a", "b", "c"], len(adata_space_rotate))
        adata_space_rotate.obs["celltype"] = adata_space_rotate.obs["celltype"].astype("category")
        # TODO(@MUCDK) use MockSolverOutput if no regression test
        ap = AlignmentProblem(adata=adata_space_rotate)
        ap = ap.prepare(batch_key="batch")
        mock_tmap = np.abs(
            rng.randn(
                len(adata_space_rotate[adata_space_rotate.obs["batch"] == "1"]),
                len(adata_space_rotate[adata_space_rotate.obs["batch"] == "2"]),
            )
        )
        ap[("1", "2")]._solution = MockSolverOutput(mock_tmap / mock_tmap.sum().sum())
        result = ap.cell_transition(
            source="1",
            target="2",
            source_groups="celltype",
            target_groups="celltype",
            forward=forward,
            normalize=normalize,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)


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

    def test_correspondence(
        self,
        adata_mapping: AnnData,
    ):
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        df = (
            MappingProblem(adataref, adatasp)
            .prepare(batch_key="batch", sc_attr={"attr": "X"})
            .spatial_correspondence(interval=[3, 4])
        )
        assert "batch" in df.columns
        np.testing.assert_array_equal(df["batch"].cat.categories, adatasp.obs["batch"].cat.categories)
        df2 = (
            MappingProblem(adataref, adatasp)
            .prepare(batch_key="batch", sc_attr={"attr": "X"})
            .spatial_correspondence(attr={"attr": "obsm", "key": "spatial"}, interval=[3, 4])
        )
        np.testing.assert_array_equal(df.index_interval.cat.categories, df2.index_interval.cat.categories)
        df3 = MappingProblem(adataref, adatasp).prepare(sc_attr={"attr": "X"}).spatial_correspondence(interval=[2, 3])
        np.testing.assert_array_equal(df3.value_interval.unique(), (2, 3))

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
            np.testing.assert_almost_equal(
                np.array(sol[k].transport_matrix), np.array(mp.solutions[k].transport_matrix), decimal=3
            )

    @pytest.mark.fast()
    @pytest.mark.parametrize("forward", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_cell_transition_pipeline(self, adata_mapping: AnnData, forward: bool, normalize: bool):
        rng = np.random.RandomState(0)
        adataref, adatasp = _adata_spatial_split(adata_mapping)
        adatasp.obs["celltype"] = rng.choice(["a", "b", "c"], len(adatasp))
        adatasp.obs["celltype"] = adatasp.obs["celltype"].astype("category")
        adataref.obs["celltype"] = rng.choice(["d", "e", "f", "g"], len(adataref))
        adataref.obs["celltype"] = adataref.obs["celltype"].astype("category")
        # TODO(@MUCDK) use MockSolverOutput if no regression test
        mp = MappingProblem(adataref, adatasp)
        mp = mp.prepare(batch_key="batch", sc_attr={"attr": "obsm", "key": "X_pca"})
        # mp = mp.solve()
        mock_tmap = np.abs(rng.randn(len(adatasp[adatasp.obs["batch"] == "1"]), len(adataref)))
        mp[("1", "ref")]._solution = MockSolverOutput(mock_tmap / np.sum(mock_tmap))

        result = mp.cell_transition(
            source="1",
            source_groups="celltype",
            target_groups="celltype",
            forward=forward,
            normalize=normalize,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 4)

    @pytest.mark.fast()
    @pytest.mark.parametrize("forward", [False,])# True])
    @pytest.mark.parametrize("mapping_mode", ["max", "sum"])
    @pytest.mark.parametrize("problem_kind", ["mapping"])
    def test_annotation_mapping(self, adata_anno: AnnData, forward: bool, mapping_mode, gt_tm_annotation):
        rng = np.random.RandomState(0)
        adataref, adatasp = _adata_spatial_split(adata_anno)
        mp = MappingProblem(adataref, adatasp)
        mp = mp.prepare(sc_attr={"attr": "obsm", "key": "X_pca"}, joint_attr={"attr": "X"})
        problem_keys = ("src", "tgt")
        assert set(mp.problems.keys()) == {problem_keys}
        mp[problem_keys].set_solution(MockSolverOutput(gt_tm_annotation.T))

        result = mp.annotation_mapping(
            mapping_mode=mapping_mode,
            annotation_label="celltype",
            source="src",
            forward=forward,
        )
        expected_result = (adataref.uns["expected_max"] if mapping_mode == "max" else adataref.uns["expected_sum"])
        assert (result["celltype"] == expected_result).all()


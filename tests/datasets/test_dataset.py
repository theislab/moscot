from types import FunctionType
from typing import Mapping
from pathlib import Path
from http.client import RemoteDisconnected
import warnings

import pytest

from anndata import AnnData, OldFormatWarning

from moscot.datasets import simulate_data
import moscot as mt


class TestDatasetsImports:
    @pytest.mark.parametrize("func", mt.datasets._datasets.__all__)
    def test_import(self, func):
        assert hasattr(mt.datasets, func), dir(mt.datasets)
        fn = getattr(mt.datasets, func)

        assert isinstance(fn, FunctionType)


# TODO(michalk8): parse the code and xfail iff server issue
class TestDatasetsDownload:
    @pytest.mark.timeout(120)
    def test_sim_align(self, tmp_path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OldFormatWarning)
            try:
                adata = mt.datasets.sim_align(tmp_path / "foo")

                assert isinstance(adata, AnnData)
                assert adata.shape == (1200, 500)
            except RemoteDisconnected as e:
                pytest.xfail(str(e))


class TestSimulateData:
    @pytest.mark.fast()
    def test_returns_adata(self):
        result = simulate_data()
        assert isinstance(result, AnnData)

    @pytest.mark.fast()
    @pytest.mark.parametrize("n_distributions", [2, 4])
    @pytest.mark.parametrize("key", ["batch", "day"])
    def test_n_distributions(self, n_distributions: int, key: str):
        adata = simulate_data(n_distributions=n_distributions, key=key)
        assert key in adata.obs.columns
        assert adata.obs[key].nunique() == n_distributions

    @pytest.mark.fast()
    @pytest.mark.parametrize("obs_to_add", [{"celltype": 2}, {"celltype": 5, "cluster": 4}])
    def test_obs_to_add(self, obs_to_add: Mapping[str, int]):
        adata = simulate_data(obs_to_add=obs_to_add)

        for colname, k in obs_to_add.items():
            assert colname in adata.obs.columns
            assert adata.obs[colname].nunique() == k

from types import FunctionType
from pathlib import Path
from http.client import RemoteDisconnected
import warnings

import pytest

from anndata import AnnData, OldFormatWarning

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

    @pytest.mark.timeout(120)
    def test_tedsim_1024(self, tmp_path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OldFormatWarning)
            try:
                adata = mt.datasets.simulation(tmp_path / "foo")

                assert isinstance(adata, AnnData)
                assert adata.shape == (1536, 500)
            except RemoteDisconnected as e:
                pytest.xfail(str(e))

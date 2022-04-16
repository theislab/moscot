import pytest

from anndata import AnnData
import scanpy as sc

_gt_temporal_adata = sc.read("tests/data/moscot_temporal_tests.h5ad")


@pytest.fixture()
def gt_temporal_adata() -> AnnData:
    return _gt_temporal_adata

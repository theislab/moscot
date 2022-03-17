import numpy.typing as npt

from anndata import AnnData


class TestSingleCompoundProblem:
    @staticmethod
    def custom_callback(adata: AnnData, adata_y: AnnData, **kwargs) -> npt.ArrayLike:
        pass

    def test_pipeline(self):
        pass

    def test_default_callback(self):
        pass

    def test_custom_callback(self):
        pass


class TestMultiCompoundProblem:
    pass

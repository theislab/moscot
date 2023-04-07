from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np
from scipy.sparse import csr_matrix

from anndata import AnnData

from moscot._types import ArrayLike
from moscot.base.output import MatrixSolverOutput
from moscot.base.problems import AnalysisMixin, CompoundProblem, OTProblem
from moscot.base.problems.compound_problem import B

Geom_t = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
RTOL = 1e-6
ATOL = 1e-6


class CompoundProblemWithMixin(CompoundProblem, AnalysisMixin):
    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return ()


class MockSolverOutput(MatrixSolverOutput):
    @property
    def cost(self) -> float:
        return 0.5

    @property
    def converged(self) -> bool:
        return True

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def potentials(self) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
        return None, None

    def _ones(self, n: int) -> ArrayLike:
        return np.ones(n)


def _make_adata(grid: ArrayLike, n: int, seed) -> List[AnnData]:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(100, 60))
    return [AnnData(X=csr_matrix(X), obsm={"spatial": grid.copy()}, dtype=X.dtype) for _ in range(n)]


def _adata_spatial_split(adata: AnnData) -> Tuple[AnnData, AnnData]:
    adata_ref = adata[adata.obs.batch == "0"].copy()
    adata_ref.obsm.pop("spatial")
    adata_sp = adata[adata.obs.batch != "0"].copy()
    return adata_ref, adata_sp


def _adata_modality_split(adata: AnnData):
    adata_tgt = adata[adata.obs.batch == "0"].copy()
    adata_src = adata[adata.obs.batch != "0"].copy()
    adata_src.obsm['emb_src'] = np.random.rand(adata_src.shape[0], 5)
    adata_tgt.obsm['emb_tgt'] = np.random.rand(adata_tgt.shape[0], 15)
    return adata_tgt, adata_src


def _make_grid(grid_size: int) -> ArrayLike:
    xlimits = ylimits = [0, 10]
    x1s = np.linspace(*xlimits, num=grid_size)
    x2s = np.linspace(*ylimits, num=grid_size)
    X1, X2 = np.meshgrid(x1s, x2s)
    return np.vstack([X1.ravel(), X2.ravel()]).T


class Problem(CompoundProblem[Any, OTProblem]):
    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return ()

from typing import Any, List, Type, Tuple, Union, Optional

from scipy.sparse import csr_matrix

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike
from moscot.problems.base import OTProblem, CompoundProblem
from moscot.solvers._output import MatrixSolverOutput
from moscot.problems.base._mixins import AnalysisMixin
from moscot.problems.base._compound_problem import B

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
    def potentials(self) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
        return None, None

    def _ones(self, n: int) -> ArrayLike:
        return np.ones(n)


class MockBaseSolverOutput:
    def __init__(self, len_a: int, len_b: int):
        rng = np.random.RandomState(42)
        self.a = rng.randn(len_a)
        self.b = rng.randn(len_b)


def _make_adata(grid: ArrayLike, n: int, seed) -> List[AnnData]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 60))
    adatas = [AnnData(X=csr_matrix(X), obsm={"spatial": grid.copy()}, dtype=X.dtype) for _ in range(n)]
    return adatas


def _adata_spatial_split(adata: AnnData) -> Tuple[AnnData, AnnData]:
    adata_ref = adata[adata.obs.batch == "0"].copy()
    adata_ref.obsm.pop("spatial")
    adata_sp = adata[adata.obs.batch != "0"].copy()
    return adata_ref, adata_sp


def _make_grid(grid_size: int) -> ArrayLike:
    xlimits = ylimits = [0, 10]
    x1s = np.linspace(*xlimits, num=grid_size)
    x2s = np.linspace(*ylimits, num=grid_size)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
    return X_orig_single


class Problem(CompoundProblem[Any, OTProblem]):
    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return ()

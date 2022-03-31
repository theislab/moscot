from typing import Optional, Any
import numpy as np
import numpy.typing as npt
from anndata import AnnData

from moscot.solvers._output import MatrixSolverOutput
from moscot.problems import MultiMarginalProblem


class TestSolverOutput(MatrixSolverOutput):
    @property
    def cost(self) -> float:
        return 0.5

    @property
    def converged(self) -> bool:
        return True

    def _ones(self, n: int) -> npt.ArrayLike:
        return np.ones(n)

class MockMultiMarginalProblem(MultiMarginalProblem):
    def _estimate_marginals(self, adata: AnnData, *, source: bool, **kwargs: Any) -> Optional[npt.ArrayLike]:
        pass


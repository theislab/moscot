from types import MappingProxyType
from typing import Any, Mapping



from anndata import AnnData

from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.problems._compound_problem import SingleCompoundProblem


class SinkhornProblem(SingleCompoundProblem):
    def __init__(self, adata: AnnData, solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any):
        super().__init__(adata, solver=SinkhornSolver(**solver_kwargs), **kwargs)

class GWProblem(SingleCompoundProblem):
    def __init__(self, adata: AnnData, solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any):
        super().__init__(adata, solver=GWSolver(**solver_kwargs), **kwargs)

class FGWProblem(SingleCompoundProblem):
    def __init__(self, adata: AnnData, solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any):
        super().__init__(adata, solver=FGWSolver(**solver_kwargs), **kwargs)

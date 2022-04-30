from types import MappingProxyType
from typing import Any, Mapping

from anndata import AnnData

from moscot._docs import d
from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.problems._compound_problem import SingleCompoundProblem


@d.dedent
class SinkhornProblem(SingleCompoundProblem):
    """
    AnnData interface for generic Optimal Transport problem

    Parameters
    ----------
    %(CompoundProblem.parameters)s

    Raises
    %(CompoundProblem.raises)s
    """

    def __init__(self, adata: AnnData, solver_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any):
        super().__init__(adata, solver=SinkhornSolver(**solver_kwargs), **kwargs)


@d.dedent
class GWProblem(SingleCompoundProblem):
    """
    AnnData interface for generic Gromov-Wasserstein problem

    Parameters
    ----------
    %(CompoundProblem.parameters)s

    Raises
    %(CompoundProblem.raises)s
    """

    def __init__(self, adata: AnnData, solver_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any):
        super().__init__(adata, solver=GWSolver(**solver_kwargs), **kwargs)


@d.dedent
class FGWProblem(SingleCompoundProblem):
    """
    AnnData interface for generic Fused Gromov-Wasserstein problem

    Parameters
    ----------
    %(CompoundProblem.parameters)s

    Raises
    %(CompoundProblem.raises)s
    """

    def __init__(self, adata: AnnData, solver_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any):
        super().__init__(adata, solver=FGWSolver(**solver_kwargs), **kwargs)

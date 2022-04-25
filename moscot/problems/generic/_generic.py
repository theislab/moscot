from types import MappingProxyType
from typing import Any, Dict, Tuple, Union, Literal, Mapping, Callable, Optional, Sequence, Type
from numbers import Number
import logging

import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData
import scanpy as sc

from moscot.problems import MultiMarginalProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._utils import beta, delta, MarkerGenes
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.problems import GeneralProblem
from moscot.mixins._time_analysis import TemporalAnalysisMixin
from moscot.solvers._tagged_array import TaggedArray
from moscot.backends.ott import SinkhornSolver, GWSolver, FGWSolver
from moscot.solvers._base_solver import ProblemKind
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
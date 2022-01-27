from typing import Any
from typing import Optional, Union, Sequence, Literal, Any, Mapping
from types import MappingProxyType
from anndata import AnnData
import numpy as np
import numpy.typing as npt
import logging

import moscot.solvers._base_solver

from numpy import typing as npt
import numpy as np
from moscot.mixins._base_analysis import AnalysisMixin, CompoundAnalysisMixin


class TemporalAnalysisMixin(CompoundAnalysisMixin):
    def push_forward(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass

    def pull_backward(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass

    def _interpolate_gex_with_ot(self,
                                 number_cells: int,
                                 adata_1: AnnData,
                                 adata_2: AnnData,
                                 solver: moscot.solvers._base_solver.BaseSolverOutput,
                                 interpolation_parameter: int = 0.5,
                                 adjust_by_growth = True
                                 ):
        #TODO: make online available
        if adjust_by_growth:
            transport_matrix = solver.transport_matrix / np.power(solver.transport_matrix.sum(axis=0), 1. - interpolation_parameter)
        else:
            transport_matrix = solver.transport_matrix
        transport_matrix = transport_matrix.flatten(order='C')
        transport_matrix_flattened = transport_matrix / transport_matrix.sum()
        choices = np.random.choice(adata_1.n_obs * adata_2.n_obs, p=transport_matrix_flattened, size=number_cells)
        return np.asarray([adata_1.X[i // adata_2.n_obs] * (1 - interpolation_parameter) + adata_2.X[i % adata_2.n_obs] * interpolation_parameter for i in choices], dtype=np.float64)

    def _interpolate_gex_randomly(self,
                                  number_cells: int,
                                  adata_1: AnnData,
                                  adata_2: AnnData,
                                  interpolation_parameter: int = 0.5,
                                  growth_rates: Optional[npt.ArrayLike] = None):

        if growth_rates is None:
            choices = np.random.choice(adata_1.n_obs * adata_2.n_obs, size=number_cells)
        else:
            outer_product = np.outer(growth_rates, np.ones(len(growth_rates)))
            outer_product_flattened = outer_product.flatten(order="C")
            outer_product_flattened /= outer_product_flattened.sum()
            choices = np.random.choice(adata_1.n_obs * adata_2.n_obs, p=outer_product_flattened, size=number_cells)

        return np.asarray([adata_1.X[i // adata_2.n_obs] * (1 - interpolation_parameter) + adata_2.X[i % adata_2.n_obs] * interpolation_parameter for i in choices], dtype=np.float64)

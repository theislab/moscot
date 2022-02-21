from typing import Any, Optional, Type
import logging
from numpy import typing as npt
import scanpy as sc
import numpy as np
from anndata import AnnData
from moscot.mixins._base_analysis import AnalysisMixin
from moscot.solvers._base_solver import BaseSolver, BaseSolverOutput
import sklearn
import ot


class TemporalAnalysisMixin(AnalysisMixin):
    def plot_trajectory(self,
                        color_key: str = None,
                        threshold: float = 1e-8,
                        probability_key: str = "pull_result",
                        show_density: bool = False,
                        **kwargs
    ):
        if color_key is None:
            color_key = self._temporal_key
        if not "neighbors" in self.adata.uns:
            sc.pp.neighbors(self.adata)
        if not "X_umap" in self.adata.obsm:
            logging.info("Calculating UMAP.")
            sc.tl.umap(self.adata, **kwargs)

        self.adata.obs["color_code_umap"] = self.adata.obs.apply(
            lambda x: x[color_key] if x[probability_key] >= threshold else np.nan, axis=1
        )
        if show_density:
            color = ["color_code_umap", probability_key]
        else:
            color = ["color_code_umap"]
        sc.pl.umap(self.adata, color=color, **kwargs)

    def validate_by_interpolation(
        self, start: Any, end: Any, intermediate: Any, interpolation_parameter: Optional[int] = None
    ):
        """
        currently this assumes that we have preprocessed data which results in the questionable assumption that
        the held out data was also used for the preprocessing (whereas it should follow the independent preprocessing
        step of WOT
        """
        if intermediate not in self.adata.obs[self._temporal_key].unique():
            raise ValueError(f"No data points corresponding to {intermediate} found in `adata.obs[{self._temporal_key}]`")
        if (start, end) not in self._problems.keys():
            logging.info(f"No transport map computed for {(start, end)}. Trying to compose transport maps.")

        if interpolation_parameter is None:
            interpolation_parameter = (intermediate - start) / (end - start)

        for subset in self._problems.keys():
            if subset[0] == start:
                source_data = self._problems[subset]._x.data
                growth_rates_source = self._problems[subset].growth_rates[-1]
                break
        for subset in self._problems.keys():
            if subset[0] == intermediate:
                intermediate_data = self._problems[subset]._x.data
                break
        for subset in self._problems.keys():
            if subset[1] == end:
                target_data = self._problems[subset]._y.data
                break
        transport_matrix = self._compute_transport_map(start=start, end=end)[1]

        gex_ot_interpolated = self._interpolate_gex_with_ot(
            len(intermediate_data), source_data, target_data, transport_matrix, interpolation_parameter
        )
        distance_gex_ot_interpolated = self._compute_wasserstein_distance(
            intermediate_data, gex_ot_interpolated
        )
        gex_randomly_interpolated = self._interpolate_gex_randomly(len(intermediate_data), source_data, target_data)
        distance_gex_randomly_interpolated = self._compute_wasserstein_distance(
            intermediate_data, gex_randomly_interpolated
        )
        gex_randomly_interpolated_growth = self._interpolate_gex_randomly(len(intermediate_data), source_data, target_data, growth_rates=growth_rates_source)
        distance_gex_randomly_interpolated_growth = self._compute_wasserstein_distance(
            intermediate_data, gex_randomly_interpolated_growth
        )

        return (
            distance_gex_ot_interpolated,
            distance_gex_randomly_interpolated,
            distance_gex_randomly_interpolated_growth,
        )
        
    def _interpolate_gex_with_ot(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        transport_matrix: npt.ArrayLike, 
        interpolation_parameter: int = 0.5,
        adjust_by_growth=True,
    ):
        # TODO(@MUCDK): make online available
        # TODO(@MUCDK): check dimensions of arrays
        if adjust_by_growth:
            transport_matrix = transport_matrix / np.power(
                transport_matrix.sum(axis=0), 1.0 - interpolation_parameter
            )
        transport_matrix_flattened = transport_matrix.flatten(order="C")
        transport_matrix_flattened /= transport_matrix_flattened.sum()
        choices = np.random.choice((len(source_data) * len(target_data))+1, p=np.concatenate((transport_matrix_flattened, np.array([1-transport_matrix_flattened.sum()])), axis=0), size=number_cells)
        #TODO(@MUCDK): think about more efficient implementation

        res = np.asarray(
            [
                source_data[i // len(target_data)] * (1 - interpolation_parameter)
                + target_data[i % len(target_data)] * interpolation_parameter
                for i in choices if i != (len(source_data) * len(target_data))+1
            ],
            dtype=np.float64,
        )
        n_to_replace = np.sum(choices==(len(source_data) * len(target_data))+1)
        rows_to_add = np.random.choice(len(res), replace=False, size=n_to_replace) # this creates a slightly biased estimator but needs to be done due to numerical errors
        return np.concatenate((res, res[rows_to_add,:]), axis=0)

    def _interpolate_gex_randomly(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        interpolation_parameter: int = 0.5,
        growth_rates: Optional[npt.ArrayLike] = None,
    ):

        if growth_rates is None:
            choices = np.random.choice(len(source_data) * len(target_data), size=number_cells)
        else:
            outer_product = np.outer(growth_rates, np.ones(len(target_data)))
            outer_product_flattened = outer_product.flatten(order="C")
            outer_product_flattened /= outer_product_flattened.sum()
            choices = np.random.choice((len(source_data) * len(target_data))+1, p=np.concatenate((outer_product_flattened, np.array([1-outer_product_flattened.sum()])), axis=0), size=number_cells)
        
        res = np.asarray(
            [
                source_data[i // len(target_data)] * (1 - interpolation_parameter)
                + target_data[i % len(target_data)] * interpolation_parameter
                for i in choices if i != (len(source_data) * len(target_data))+1
            ],
            dtype=np.float64,
        )
        n_to_replace = np.sum(choices==(len(source_data) * len(target_data))+1)
        rows_to_add = np.random.choice(len(res), replace=False, size=n_to_replace) # this creates a slightly biased estimator but needs to be done due to numerical errors
        return np.concatenate((res, res[rows_to_add,:]), axis=0)

    #TODO(@MUCDK) possibly offer two alternatives, once exact EMD with POT backend and once approximate, faster with same solver as used for original problems
    def _compute_wasserstein_distance(
        self,
        point_cloud_1: npt.ArrayLike,
        point_cloud_2: npt.ArrayLike,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        **kwargs: Any
    ):
        cost_matrix = sklearn.metrics.pairwise.pairwise_distances(point_cloud_1, Y=point_cloud_2, metric='sqeuclidean', n_jobs=-1)
        if a is None:
            a = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
        if b is None:
            b = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]
        return np.sqrt(ot.emd2(a, b, cost_matrix, **kwargs))


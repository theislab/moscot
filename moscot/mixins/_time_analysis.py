from typing import Any, Dict, Optional
from numbers import Number
import logging
import itertools

import ot
import sklearn

from numpy import typing as npt
import numpy as np

from anndata import AnnData

from moscot.mixins._base_analysis import AnalysisMixin


class TemporalAnalysisMixin(AnalysisMixin):
    def validate_by_interpolation(
        self,
        start: Any,
        end: Any,
        intermediate: Any,
        interpolation_parameter: Optional[int] = None,
        val_ot: bool = True,
        val_random: bool = True,
        val_random_with_growth: bool = True,
        val_source_to_intermediate: bool = True,
        val_intermediate_to_target: bool = True,
        batch_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Number]:
        """
        currently this assumes that we have preprocessed data which results in the questionable assumption that
        the held out data was also used for the preprocessing (whereas it should follow the independent preprocessing
        step of WOT
        """
        if intermediate not in self.adata.obs[self._temporal_key].unique():
            raise ValueError(
                f"No data points corresponding to {intermediate} found in `adata.obs[{self._temporal_key}]`"
            )
        if (start, end) not in self._problems.keys():
            logging.info(f"No transport map computed for {(start, end)}. Trying to compose transport maps.")

        if interpolation_parameter is None:
            interpolation_parameter = (intermediate - start) / (end - start)

        for (start_, end_) in self._problems.keys():
            if start_ == start:
                source_data = self._problems[(start_, end_)]._x.data
                growth_rates_source = self._problems[(start_, end_)].growth_rates[-1]
                break
        else:
            raise ValueError(f"No data found for time point {start}")
        for (start_, end_) in self._problems.keys():
            if start_ == intermediate:
                intermediate_data = self._problems[(start_, end_)]._x.data
                intermediate_adata = self._problems[(start_, end_)].adata
                break
        else:
            raise ValueError(f"No data found for time point {intermediate}")
        for (start_, end_) in self._problems.keys():
            if end_ == end:
                target_data = self._problems[(start_, end_)]._y.data
                break
        else:
            raise ValueError(f"No data found for time point {end}")
        transport_matrix = self._compute_transport_map(start=start, end=end)

        result = {}
        if val_ot:
            gex_ot_interpolated = self._interpolate_gex_with_ot(
                len(intermediate_data), source_data, target_data, transport_matrix
            )
            result["ot"] = self._compute_wasserstein_distance(intermediate_data, gex_ot_interpolated, **kwargs)

        if val_random:
            gex_randomly_interpolated = self._interpolate_gex_randomly(len(intermediate_data), source_data, target_data)
            result["random"] = self._compute_wasserstein_distance(
                intermediate_data, gex_randomly_interpolated, **kwargs
            )

        if val_random_with_growth:
            gex_randomly_interpolated_growth = self._interpolate_gex_randomly(
                len(intermediate_data), source_data, target_data, growth_rates=growth_rates_source
            )
            result["random_with_growth"] = self._compute_wasserstein_distance(
                intermediate_data, gex_randomly_interpolated_growth, **kwargs
            )

        if val_source_to_intermediate:
            result["source_to_intermediate"] = self._compute_wasserstein_distance(
                source_data, intermediate_data, **kwargs
            )

        if val_intermediate_to_target:
            result["intermediate_to_target"] = self._compute_wasserstein_distance(
                intermediate_data, target_data, **kwargs
            )
        if batch_key is not None:

            result["batches"] = self._compute_distance_between_batches(
                intermediate_adata, intermediate_data, batch_key, **kwargs
            )

        return result

    def _interpolate_gex_with_ot(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        transport_matrix: npt.ArrayLike,
        interpolation_parameter: int = 0.5,
        adjust_by_growth=True,
    ) -> npt.ArrayLike:
        # TODO(@MUCDK): make online available
        # TODO(@MUCDK): check dimensions of arrays
        if adjust_by_growth:
            transport_matrix = transport_matrix / np.power(transport_matrix.sum(axis=0), 1.0 - interpolation_parameter)
        transport_matrix_flattened = transport_matrix.flatten(order="C")
        transport_matrix_flattened /= transport_matrix_flattened.sum()
        choices = np.random.choice(
            (len(source_data) * len(target_data)) + 1,
            p=np.concatenate((transport_matrix_flattened, np.array([1 - transport_matrix_flattened.sum()])), axis=0),
            size=number_cells,
        )
        # TODO(@MUCDK): think about more efficient implementation

        res = np.asarray(
            [
                source_data[i // len(target_data)] * (1 - interpolation_parameter)
                + target_data[i % len(target_data)] * interpolation_parameter
                for i in choices
                if i != (len(source_data) * len(target_data)) + 1
            ],
            dtype=np.float64,
        )
        n_to_replace = np.sum(choices == (len(source_data) * len(target_data)) + 1)
        rows_to_add = np.random.choice(
            len(res), replace=False, size=n_to_replace
        )  # this creates a slightly biased estimator but needs to be done due to numerical errors
        return np.concatenate((res, res[rows_to_add, :]), axis=0)

    def _interpolate_gex_randomly(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        interpolation_parameter: int = 0.5,
        growth_rates: Optional[npt.ArrayLike] = None,
    ) -> npt.ArrayLike:

        if growth_rates is None:
            choices = np.random.choice(len(source_data) * len(target_data), size=number_cells)
        else:
            outer_product = np.outer(growth_rates ** interpolation_parameter, np.ones(len(target_data)))
            outer_product_flattened = outer_product.flatten(order="C")
            outer_product_flattened /= outer_product_flattened.sum()
            choices = np.random.choice(
                (len(source_data) * len(target_data)) + 1,
                p=np.concatenate((outer_product_flattened, np.array([1 - outer_product_flattened.sum()])), axis=0),
                size=number_cells,
            )

        res = np.asarray(
            [
                source_data[i // len(target_data)] * (1 - interpolation_parameter)
                + target_data[i % len(target_data)] * interpolation_parameter
                for i in choices
                if i != (len(source_data) * len(target_data)) + 1
            ],
            dtype=np.float64,
        )
        n_to_replace = np.sum(choices == (len(source_data) * len(target_data)) + 1)
        rows_to_add = np.random.choice(
            len(res), replace=False, size=n_to_replace
        )  # this creates a slightly biased estimator but needs to be done due to numerical errors
        return np.concatenate((res, res[rows_to_add, :]), axis=0)

    def _compute_distance_between_batches(
        self, adata: AnnData, data: npt.ArrayLike, batch_key: str, **kwargs: Any
    ) -> Number:
        assert len(adata) == len(data), "TODO: wrong shapes"
        dist = []
        for batch_1, batch_2 in itertools.combinations(adata.obs[batch_key].unique(), 2):
            dist.append(
                self._compute_wasserstein_distance(
                    data[adata.obs[batch_key == batch_1]], data[adata.obs[batch_key == batch_2]], **kwargs
                )
            )
        return np.mean(dist)

    # TODO(@MUCDK) possibly offer two alternatives, once exact EMD with POT backend and once approximate, faster with same solver as used for original problems
    def _compute_wasserstein_distance(
        self,
        point_cloud_1: npt.ArrayLike,
        point_cloud_2: npt.ArrayLike,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        **kwargs: Any,
    ) -> Number:
        cost_matrix = sklearn.metrics.pairwise.pairwise_distances(
            point_cloud_1, Y=point_cloud_2, metric="sqeuclidean", n_jobs=-1
        )
        if a is None:
            a = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
        if b is None:
            b = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]
        return np.sqrt(ot.emd2(a, b, cost_matrix, **kwargs))

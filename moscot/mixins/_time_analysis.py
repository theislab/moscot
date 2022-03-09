from types import MappingProxyType
from typing import Any, Dict, Mapping, Callable, Optional
from numbers import Number
from functools import partial
import logging

from sklearn.metrics.pairwise import pairwise_distances
import ot

from numpy import typing as npt
import numpy as np

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
        n_interpolated_cells: Optional[int] = None,
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
                growth_rates_source = self._problems[(start_, end_)].growth_rates[:, -1]
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

        if n_interpolated_cells is None:
            n_interpolated_cells = len(intermediate_data)

        result = {}
        if val_ot:
            gex_ot_interpolated = self._interpolate_gex_with_ot(
                n_interpolated_cells, source_data, target_data, start, end, interpolation_parameter
            )
            result["ot"] = self._compute_wasserstein_distance(intermediate_data, gex_ot_interpolated, **kwargs)

        if val_random:
            gex_randomly_interpolated = self._interpolate_gex_randomly(
                n_interpolated_cells, source_data, target_data, interpolation_parameter, **kwargs
            )
            result["random"] = self._compute_wasserstein_distance(
                intermediate_data, gex_randomly_interpolated, **kwargs
            )

        if val_random_with_growth:
            gex_randomly_interpolated_growth = self._interpolate_gex_randomly(
                len(intermediate_data), source_data, target_data, growth_rates=growth_rates_source, **kwargs
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
            if batch_key not in self.adata.obs.columns:
                raise ValueError(f"{batch_key} not found in `adata.obs.columns`")
            result["batches"] = self._compute_distance_between_batches(
                intermediate_adata, intermediate_data, batch_key, **kwargs
            )

        return result

    # TODO(@MUCDK) possibly offer two alternatives, once exact EMD with POT backend and once approximate, faster with same solver as used for original problems
    def _compute_wasserstein_distance(
        self,
        point_cloud_1: npt.ArrayLike,
        point_cloud_2: npt.ArrayLike,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        numItermax: Optional[int] = 1e6,
        **kwargs: Any,
    ) -> Number:
        cost_matrix = pairwise_distances(point_cloud_1, Y=point_cloud_2, metric="sqeuclidean", n_jobs=-1)
        if a is None:
            a = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
        if b is None:
            b = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]
        return np.sqrt(ot.emd2(a, b, cost_matrix, numItermax=numItermax, **kwargs))

    def _interpolate_gex_with_ot(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        start: Any,
        end: Any,
        interpolation_parameter: float = 0.5,
        adjust_by_growth: bool = True,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> npt.ArrayLike:
        def mappable_choice(a: int, kwargs: Mapping[str, Any] = MappingProxyType({})) -> Callable[[Any], npt.ArrayLike]:
            return partial(np.random.choice, a=a, replace=True)(**kwargs)

        mass = np.ones(len(target_data))
        if adjust_by_growth:
            col_sums = (
                np.array(
                    np.squeeze(
                        self.push(
                            start=start,
                            end=end,
                            normalize=True,
                            scale_by_marginals=False,
                            plans={(start, end): [(start, end)]},
                        )
                    )
                )
                + 1e-9
            )
            mass = mass / np.power(col_sums, 1 - interpolation_parameter)
        row_probability = np.array(
            np.squeeze(
                self.pull(
                    start=start,
                    end=end,
                    data=mass,
                    normalize=True,
                    scale_by_marginals=False,
                    plans={(start, end): [(start, end)]},
                )
            )
        )

        p = row_probability / row_probability.sum()
        rows_sampled = np.random.choice(len(source_data), p=p, size=number_cells)
        rows, counts = np.unique(rows_sampled, return_counts=True)
        result = np.zeros((number_cells, source_data.shape[1]))
        current_index = 0
        for batch in range(int(np.floor(len(rows) / batch_size))):
            rows_batch = rows[batch * batch_size : (batch + 1) * batch_size]
            counts_batch = counts[batch * batch_size : (batch + 1) * batch_size]
            data = np.zeros((len(source_data), batch_size))
            data[rows_batch, range(batch_size)] = 1
            col_p_given_row = np.array(
                np.squeeze(
                    self.push(
                        start=start,
                        end=end,
                        data=data,
                        normalize=True,
                        scale_by_marginals=False,
                        plans={(start, end): [(start, end)]},
                    )
                )
            )
            if adjust_by_growth:
                col_p_given_row = col_p_given_row / col_sums[:, None]
            kwargs_list = [
                dict(size=counts_batch[i], p=col_p_given_row[:, i] / col_p_given_row[:, i].sum())
                for i in range(batch_size)
            ]
            cols_sampled = list(map(mappable_choice, [len(target_data)] * len(kwargs_list), kwargs_list))
            updated_index = current_index + np.sum(counts_batch)
            result[current_index:updated_index, :] = (
                source_data[np.repeat(rows_batch, counts_batch), :] * (1 - interpolation_parameter)
                + target_data[np.hstack(cols_sampled), :] * interpolation_parameter
            )
            current_index = updated_index
        remaining_batch_size = len(rows) % batch_size
        rows_batch = rows[(batch + 1) * batch_size :]
        counts_batch = counts[(batch + 1) * batch_size :]
        data = np.zeros((len(source_data), remaining_batch_size))
        data[rows_batch, range(remaining_batch_size)] = 1
        col_p_given_row = np.array(
            np.squeeze(
                self.push(
                    start=start,
                    end=end,
                    data=data,
                    normalize=True,
                    scale_by_marginals=False,
                    plans={(start, end): [(start, end)]},
                )
            )
        )
        if adjust_by_growth:
            col_p_given_row = col_p_given_row / col_sums[:, None]
        kwargs_list = [
            dict(size=counts_batch[i], p=col_p_given_row[:, i] / col_p_given_row[:, i].sum())
            for i in range(remaining_batch_size)
        ]
        cols_sampled = list(map(mappable_choice, [len(target_data)] * len(kwargs_list), kwargs_list))
        updated_index = current_index + np.sum(counts_batch)
        result[current_index:updated_index, :] = (
            source_data[np.repeat(rows_batch, counts_batch), :] * (1 - interpolation_parameter)
            + target_data[np.hstack(cols_sampled), :] * interpolation_parameter
        )

        return result

    def _interpolate_gex_randomly(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        interpolation_parameter: int = 0.5,
        growth_rates: Optional[npt.ArrayLike] = None,
        **kwargs: Any,
    ) -> npt.ArrayLike:
        if growth_rates is None:
            row_probability = np.ones(len(source_data)).astype("float64")
        else:
            row_probability = growth_rates ** (1 - interpolation_parameter)
        result = (
            source_data[
                np.random.choice(len(source_data), size=number_cells, p=row_probability / np.sum(row_probability)), :
            ]
            * (1 - interpolation_parameter)
            + target_data[np.random.choice(len(target_data), size=number_cells), :] * interpolation_parameter
        )
        return result

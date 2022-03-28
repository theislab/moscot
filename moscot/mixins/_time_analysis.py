from typing import Any, Tuple, Union, Optional
from numbers import Number
import itertools

from sklearn.metrics.pairwise import pairwise_distances
import ot

from numpy import typing as npt
import numpy as np

from anndata import AnnData

from moscot.mixins._base_analysis import AnalysisMixin


class TemporalAnalysisMixin(AnalysisMixin):
    def _get_data(
        self, start: Number, intermediate: Optional[Number] = None, end: Optional[Number] = None
    ) -> Tuple[Union[npt.ArrayLike, AnnData]]:
        for (start_, end_) in self._problems.keys():
            if start_ == start:
                source_data = self._problems[(start_, end_)]._x.data
                growth_rates_source = self._problems[(start_, end_)].growth_rates[:, -1]
                if intermediate is None:
                    return source_data, growth_rates_source
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

        return source_data, growth_rates_source, intermediate_data, intermediate_adata, target_data

    def _get_interpolation_parameter(
        interpolation_parameter: Number, start: Number, intermediate: Number, end: Number
    ) -> Number:
        return (
            interpolation_parameter if interpolation_parameter is not None else (intermediate - start) / (end - start)
        )

    def _get_n_interpolated_cells(n: Number, intermediate_data: npt.ArrayLike) -> Number:
        return n if n is not None else len(intermediate_data)

    def compute_interpolated_distance(
        self,
        start: Number,
        intermediate: Number,
        end: Number,
        interpolation_parameter: Optional[int] = None,
        n_interpolated_cells: Optional[int] = None,
        account_for_unbalancedness: bool = False,
        batch_size: int = 256,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Number:

        source_data, _, intermediate_data, _, target_data = self._get_data(start, intermediate, end)
        interpolation_parameter = self._get_interpolation_parameter(interpolation_parameter, start, intermediate, end)
        n_interpolated_cells = self._get_n_interpolated_cells(n_interpolated_cells, intermediate_data)
        interpolation = self._interpolate_gex_with_ot(
            n_interpolated_cells,
            source_data,
            target_data,
            start,
            end,
            interpolation_parameter,
            account_for_unbalancedness,
            batch_size=batch_size,
            seed=seed,
        )
        return self._compute_wasserstein_distance(intermediate_data, interpolation, **kwargs)

    def compute_random_distance(
        self,
        start: Number,
        intermediate: Number,
        end: Number,
        interpolation_parameter: Optional[int] = None,
        n_interpolated_cells: Optional[int] = None,
        account_for_unbalancedness: bool = False,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Number:
        source_data, growth_rates_source, intermediate_data, _, target_data = self._get_data(start, intermediate, end)
        interpolation_parameter = self._get_interpolation_parameter(interpolation_parameter, start, intermediate, end)
        n_interpolated_cells = self._get_n_interpolated_cells(n_interpolated_cells, intermediate_data)

        growth_rates = growth_rates_source if account_for_unbalancedness else None
        random_interpolation = self._interpolate_gex_randomly(
            n_interpolated_cells,
            source_data,
            target_data,
            interpolation_parameter,
            growth_rates=growth_rates,
            seed=seed,
        )
        return self._compute_wasserstein_distance(intermediate_data, random_interpolation, **kwargs)

    def compute_time_point_distances(
        self, start: Number, intermediate: Number, end: Number, **kwargs: Any
    ) -> Tuple[Number]:
        source_data, _, intermediate_data, _, target_data = self._get_data(start, intermediate, end)

        distance_source_intermediate = self._compute_wasserstein_distance(source_data, intermediate_data, **kwargs)
        distance_intermediate_target = self._compute_wasserstein_distance(intermediate_data, target_data, **kwargs)

        return distance_source_intermediate, distance_intermediate_target

    def compute_batch_distances(self, time: Number, batch_key: str, **kwargs: Any):
        data, adata = self._get_data(time)
        return self._compute_distance_between_batches(adata, data, batch_key, **kwargs)

    # TODO(@MUCDK) possibly offer two alternatives, once exact EMD with POT backend and once approximate, faster with same solver as used for original problems
    def _compute_wasserstein_distance(
        self,
        point_cloud_1: npt.ArrayLike,
        point_cloud_2: npt.ArrayLike,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        **kwargs: Any,
    ) -> Number:
        cost_matrix = pairwise_distances(point_cloud_1, Y=point_cloud_2, metric="sqeuclidean", n_jobs=-1)
        a = [] if a is None else a
        b = [] if b is None else b
        return np.sqrt(ot.emd2(a, b, cost_matrix, **kwargs))

    def _interpolate_gex_with_ot(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        start: Number,
        end: Number,
        interpolation_parameter: float = 0.5,
        account_for_unbalancedness: bool = True,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ) -> npt.ArrayLike:

        rows_sampled, cols_sampled = self._sample_from_tmap(
            start=start,
            end=end,
            n_samples=number_cells,
            source_dim=len(source_data),
            target_dim=len(target_data),
            batch_size=batch_size,
            account_for_unbalancedness=account_for_unbalancedness,
            interpolation_parameter=interpolation_parameter,
            seed=seed,
        )
        return (
            source_data[np.repeat(rows_sampled, [len(col) for col in cols_sampled]), :] * (1 - interpolation_parameter)
            + target_data[np.hstack(cols_sampled), :] * interpolation_parameter
        )

    def _interpolate_gex_randomly(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        interpolation_parameter: int = 0.5,
        growth_rates: Optional[npt.ArrayLike] = None,
        seed: Optional[int] = None,
    ) -> npt.ArrayLike:
        rng = np.random.RandomState(seed)
        if growth_rates is None:
            row_probability = np.ones(len(source_data))
        else:
            row_probability = growth_rates ** (1 - interpolation_parameter)
        row_probability /= np.sum(row_probability)
        result = (
            source_data[rng.choice(len(source_data), size=number_cells, p=row_probability), :]
            * (1 - interpolation_parameter)
            + target_data[rng.choice(len(target_data), size=number_cells), :] * interpolation_parameter
        )
        return result

    def _compute_distance_between_batches(
        self, adata: AnnData, data: npt.ArrayLike, batch_key: str, **kwargs: Any
    ) -> Number:
        assert len(adata) == len(data), "TODO: wrong shapes"
        dist = []
        for batch_1, batch_2 in itertools.combinations(adata.obs[batch_key].unique(), 2):
            dist.append(
                self._compute_wasserstein_distance(
                    data[(adata.obs[batch_key] == batch_1).values, :],
                    data[(adata.obs[batch_key] == batch_2).values, :],
                    **kwargs,
                )
            )
        return np.mean(dist)

from typing import Any, Tuple, Union, Mapping, Optional, Sequence
from numbers import Number
import logging
import itertools

from sklearn.metrics.pairwise import pairwise_distances
import ot
import pandas as pd

from numpy import typing as npt
import numpy as np

from anndata import AnnData

from moscot.mixins._base_analysis import AnalysisMixin


class TemporalAnalysisMixin(AnalysisMixin):
    def cell_transition(
        self,
        start: Any,
        end: Any,
        early_cells: Union[str, Mapping[str, Sequence[Any]]],
        late_cells: Union[str, Mapping[str, Sequence[Any]]],
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        **kwargs: Any,
    ) -> pd.DataFrame:
        _early_cells_key, _early_cells = self._validate_args_cell_transition(early_cells)
        _late_cells_key, _late_cells = self._validate_args_cell_transition(late_cells)

        transition_table = pd.DataFrame(
            np.zeros((len(_early_cells), len(_late_cells))), index=_early_cells, columns=_late_cells
        )

        df_late = self.adata[self.adata.obs[self._temporal_key] == end].obs[[_late_cells_key]].copy()
        df_early = self.adata[self.adata.obs[self._temporal_key] == start].obs[[_early_cells_key]].copy()

        if forward:
            _early_cells_present = set(_early_cells).intersection(set(df_early[_early_cells_key].unique()))
            for subset in _early_cells:
                if subset not in _early_cells_present:
                    transition_table.loc[subset, :] = np.nan
                    continue
                try:
                    result = self.push(
                        start=start,
                        end=end,
                        data=_early_cells_key,
                        subset=subset,
                        normalize=True,
                        return_all=False,
                        scale_by_marginals=True,
                        **kwargs,
                    )
                except ValueError as e:
                    if "no mass" in str(e):  # TODO: adapt
                        logging.info(
                            f"No data points corresponding to {subset} found in `adata.obs[groups_key]` for {start}"
                        )
                        result = np.nan
                    else:
                        raise
                df_late.loc[:, "distribution"] = result / np.sum(result)
                target_cell_dist = df_late[df_late[_late_cells_key].isin(_late_cells)].groupby(_late_cells_key).sum()
                transition_table.loc[subset, :] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else 0
                    for cell_type in _late_cells
                ]
            return transition_table
        _late_cells_present = set(_late_cells).intersection(set(df_late[_late_cells_key].unique()))
        for subset in _late_cells:
            if subset not in _late_cells_present:
                transition_table.loc[:, subset] = np.nan
                continue
            try:
                result = self.pull(
                    start=start,
                    end=end,
                    data=_late_cells_key,
                    subset=subset,
                    normalize=True,
                    return_all=False,
                    scale_by_marginals=True,
                    **kwargs,
                )
            except ValueError as e:
                if "no mass" in str(e):  # TODO: adapt
                    logging.info(f"No data points corresponding to {subset} found in `adata.obs[groups_key]` for {end}")
                    result = np.nan
                else:
                    raise
            df_early.loc[:, "distribution"] = result / np.sum(result)
            target_cell_dist = df_early[df_early[_early_cells_key].isin(_early_cells)].groupby(_early_cells_key).sum()
            transition_table.loc[:, subset] = [
                target_cell_dist.loc[cell_type, "distribution"]
                if cell_type in target_cell_dist.distribution.index
                else 0
                for cell_type in _early_cells
            ]
        return transition_table

    def _validate_args_cell_transition(
        self, arg: Union[str, Mapping[str, Sequence[Any]]]
    ) -> Tuple[Union[str, Sequence], Sequence]:
        if isinstance(arg, str):
            if not hasattr(self.adata.obs[arg], "cat"):
                raise ValueError(f"The column `{arg}` in `adata.obs` must be of categorical dtype")
            return arg, list(self.adata.obs[arg].unique())
        _key, _arg = arg.keys(), arg.values()
        if not hasattr(self.adata.obs[_key], "cat"):
            raise ValueError(f"The column `{_key}` in `adata.obs` must be of categorical dtype")
        if len(_key) != 1:
            raise ValueError("The data can only be filtered according to one column of `adata.obs`")
        if not set(_arg).isin(set(self.adata.obs[_key].unique())):
            raise ValueError(f"Not all values {_arg} could be found in column {_key}")
        return _key, _arg

    def _get_data(
        self,
        key: Number,
        intermediate: Optional[Number] = None,
        end: Optional[Number] = None,
        *,
        only_start: bool = False,
    ) -> Tuple[Union[npt.ArrayLike, AnnData], ...]:
        for (start_, end_) in self._problems.keys():
            if start_ == key:
                source_data = self._problems[(start_, end_)]._x.data
                if only_start:
                    return source_data, self._problems[(start_, end_)].adata
                growth_rates_source = self._problems[(start_, end_)].growth_rates[:, -1]
                break
        else:
            raise ValueError(f"No data found for time point {key}")
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
        source_data, _, intermediate_data, _, target_data = self._get_data(start, intermediate, end, only_start=False)
        interpolation_parameter = self._get_interp_param(interpolation_parameter, start, intermediate, end)
        n_interpolated_cells = n_interpolated_cells if n_interpolated_cells is not None else len(intermediate_data)
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
        source_data, growth_rates_source, intermediate_data, _, target_data = self._get_data(
            start, intermediate, end, only_start=False
        )
        interpolation_parameter = self._get_interp_param(interpolation_parameter, start, intermediate, end)
        n_interpolated_cells = n_interpolated_cells if n_interpolated_cells is not None else len(intermediate_data)

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
    ) -> Tuple[Number, Number]:
        source_data, _, intermediate_data, _, target_data = self._get_data(start, intermediate, end, only_start=False)

        distance_source_intermediate = self._compute_wasserstein_distance(source_data, intermediate_data, **kwargs)
        distance_intermediate_target = self._compute_wasserstein_distance(intermediate_data, target_data, **kwargs)

        return distance_source_intermediate, distance_intermediate_target

    def compute_batch_distances(self, time: Number, batch_key: str, **kwargs: Any) -> float:
        data, adata = self._get_data(time, only_start=True)
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

    def _get_interp_param(interpolation_parameter: Number, start: Number, intermediate: Number, end: Number) -> Number:
        if 0 > interpolation_parameter or interpolation_parameter > 1:
            raise ValueError("TODO: interpolation parameter must be in [0,1].")
        if start >= intermediate:
            raise ValueError("TODO: expected start < intermediate")
        if intermediate >= end:
            raise ValueError("TODO: expected intermediate < end")
        return (
            interpolation_parameter if interpolation_parameter is not None else (intermediate - start) / (end - start)
        )

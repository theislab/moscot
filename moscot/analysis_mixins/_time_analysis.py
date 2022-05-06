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

from moscot.analysis_mixins._base_analysis import AnalysisMixin


class TemporalAnalysisMixin(AnalysisMixin):
    _TEMPORAL_KEY: Optional[str] = None

    def cell_transition(
        self,
        start: Any,
        end: Any,
        early_cells: Union[str, Mapping[str, Sequence[Any]]],
        late_cells: Union[str, Mapping[str, Sequence[Any]]],
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        start
            Time point corresponding to the early distribution.
        end
            Time point corresponding to the late distribution.
        early_cells
            Can be one of the following
                - if `early_cells` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in `anndata.AnnData.obs` ``['{early_cells}']``
                - if `early_cells` is of type `Mapping` its `key` should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its `value` to a subset of categories present in
                  `anndata.AnnData.obs` ``['{early_cells.keys()[0]}']``
        late_cells
            Can be one of the following
                - if `late_cells` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in `anndata.AnnData.obs` ``['{late_cells}']``
                - if `late_cells` is of type `Mapping` its `key` should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its `value` to a subset of categories present in
                  `anndata.AnnData.obs` ``['{late_cells.keys()[0]}']``

        Returns
        -------
        Transition matrix of groups between time points.
        """
        _early_cells_key, _early_cells = self._validate_args_cell_transition(early_cells)
        _late_cells_key, _late_cells = self._validate_args_cell_transition(late_cells)

        transition_table = pd.DataFrame(
            np.zeros((len(_early_cells), len(_late_cells))), index=_early_cells, columns=_late_cells
        )

        df_late = self.adata[self.adata.obs[self.temporal_key] == end].obs[[_late_cells_key]].copy()
        df_early = self.adata[self.adata.obs[self.temporal_key] == start].obs[[_early_cells_key]].copy()
        df_late["distribution"] = np.nan
        df_early["distribution"] = np.nan

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
                        scale_by_marginals=False,
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
                df_late.loc[:, "distribution"] = result
                target_cell_dist = df_late[df_late[_late_cells_key].isin(_late_cells)].groupby(_late_cells_key).sum()
                target_cell_dist /= target_cell_dist.sum()
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
                    scale_by_marginals=False,
                    **kwargs,
                )
            except ValueError as e:
                if "no mass" in str(e):  # TODO: adapt
                    logging.info(f"No data points corresponding to {subset} found in `adata.obs[groups_key]` for {end}")
                    result = np.nan
                else:
                    raise
            df_early.loc[:, "distribution"] = result
            target_cell_dist = df_early[df_early[_early_cells_key].isin(_early_cells)].groupby(_early_cells_key).sum()
            target_cell_dist /= target_cell_dist.sum()
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
            if self.adata.obs[arg].isnull().sum() > 0:
                raise ValueError(
                    f"The column `{arg}` in `adata.obs` contains NaN values. Please fill them with another value."
                )
            return arg, list(self.adata.obs[arg].unique())
        if len(arg) > 1:
            raise ValueError(
                f"The length of the dictionary is {len(arg)} but should be 1 as the data can only be filtered "
                f"according to one column of `adata.obs`"
            )
        _key, _arg = list(arg.keys())[0], list(arg.values())[0]
        if not hasattr(self.adata.obs[_key], "cat"):
            raise ValueError(f"The column `{_key}` in `adata.obs` must be of categorical dtype")
        if not set(_arg).issubset(set(self.adata.obs[_key].unique())):
            raise ValueError(f"Not all values {_arg} could be found in column {_key}")
        if self.adata.obs[_key].isnull().sum() > 0:
            raise ValueError(
                f"The column `{_key}` in `adata.obs` contains NaN values. Please fill them with another value."
            )
        return _key, _arg

    def _get_data(
        self,
        key: Number,
        intermediate: Optional[Number] = None,
        end: Optional[Number] = None,
        *,
        only_start: bool = False,
    ) -> Tuple[Union[npt.ArrayLike, AnnData], ...]:
        # TODO(michalk8): refactor me
        for (start_, end_) in self._problems.keys():
            if self._problems[(start_, end_)].xy[0].tag != "point_cloud":
                raise ValueError(
                    f"TODO: This method requires the data to be stored as point_clouds. It is currently stored "
                    "as {self._problems[(start_, end_)].xy[0].tag}"
                )
            if start_ == key:
                source_data = self._problems[(start_, end_)].xy[0].data
                if only_start:
                    return source_data, self._problems[(start_, end_)].adata
                growth_rates_source = self._problems[(start_, end_)].growth_rates[:, -1]
                break
        else:
            raise ValueError(f"No data found for time point {key}")
        for (start_, end_) in self._problems.keys():
            if start_ == intermediate:
                intermediate_data = self._problems[(start_, end_)].xy[0].data
                intermediate_adata = self._problems[(start_, end_)].adata
                break
        else:
            raise ValueError(f"No data found for time point {intermediate}")
        for (start_, end_) in self._problems.keys():
            if end_ == end:
                target_data = self._problems[(start_, end_)].xy[1].data
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
        """
        Compute the Wasserstein distance between the OT-interpolated distribution and the true cell distribution.

        This is a validation method which interpolates the cell distributions corresponding to `start` and `end`
        leveraging the OT coupling to obtain an approximation of the cell distribution at time point `intermediate`.
        Therefore, the Wasserstein distance between the interpolated and the real distribution is computed.

        It is recommended to compare the Wasserstein distance to the ones obtained by
        :meth:`compute_time_point_distances`,
        :meth:`compute_random_distance`, and
        :meth:`compute_time_point_distance`.

        This method does not instantiate the transport matrix if the solver output does not.

        TODO: link to notebook


        Parameters
        ----------
        start
            Time point corresponding to the early distribution.
        intermediate
            Time point corresponding to the intermediate distribution which is to be interpolated.
        end
            Time point corresponding to the late distribution.
        interpolation_parameter
            Interpolation parameter determining the weight of the source and the target distribution. If `None`
            it is linearly inferred from `source`, `intermediate`, and `target`.
        n_interpolated_cells
            Number of generated interpolated cell. If `None` the number of data points in the `intermediate`
            distribution is taken.
        account_for_unbalancedness
            If `True` unbalancedness is accounted for by assuming exponential growth and death of cells.
        batch_size
            Number of cells simultaneously generated by interpolation.
        seed
            Random seed for sampling from the transport matrix.
        kwargs
            Keyword arguments for computing the Wasserstein distance (TODO make that function public?)

        Returns
        -------
        Wasserstein distance between OT-based interpolated distribution and the true cell distribution.
        """
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
        """
        Compute the Wasserstein distance of a randomly interpolated cell distribution and the true cell distribution.

        This method interpolates the cell trajectories at the `intermediate` time point using a random coupling and
        computes the distance to the true cell distribution.

        TODO: link to notebook

        Parameters
        ----------
        start
            Time point corresponding to the early distribution.
        intermediate
            Time point corresponding to the intermediate distribution which is to be interpolated.
        end
            Time point corresponding to the late distribution.
        interpolation_parameter
            Interpolation parameter determining the weight of the source and the target distribution. If `None`
            it is linearly inferred from `source`, `intermediate`, and `target`.
        n_interpolated_cells
            Number of generated interpolated cell. If `None` the number of data points in the `intermediate`
            distribution is taken.
        account_for_unbalancedness
            If `True` unbalancedness is accounted for by assuming exponential growth and death of cells.
        seed
            Random seed for generating randomly interpolated cells.
        kwargs
            Keyword arguments for computing the Wasserstein distance (TODO make that function public?)

        Returns
        -------
        The Wasserstein distance between a randomly interpolated cell distribution and the true cell distribution.
        """
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
        """
        Compute the Wasserstein distance of cell distributions between time points.

        This method computes the Wasserstein distance between the cell distribution corresponding to `start` and `
        intermediate` and `intermediate` and `end`, respectively.

        TODO: link to notebook

        Parameters
        ----------
        start
            Time point corresponding to the early distribution.
        intermediate
            Time point corresponding to the intermediate distribution.
        end
            Time point corresponding to the late distribution.
        kwargs
            Keyword arguments for computing the Wasserstein distance (TODO make that function public?).
        """
        source_data, _, intermediate_data, _, target_data = self._get_data(start, intermediate, end, only_start=False)

        distance_source_intermediate = self._compute_wasserstein_distance(source_data, intermediate_data, **kwargs)
        distance_intermediate_target = self._compute_wasserstein_distance(intermediate_data, target_data, **kwargs)

        return distance_source_intermediate, distance_intermediate_target

    def compute_batch_distances(self, time: Number, batch_key: str, **kwargs: Any) -> float:
        """
        Compute the mean Wasserstein distance between batches of a distribution corresponding to one time point.

        Parameters
        ----------
        time
            Time point corresponding to the cell distribution which to compute the batch distances within.
        batch_key
            Key in :attr:`anndata.AnnData.obs` storing which batch each cell belongs to.
        kwargs
            Keyword arguments for computing the Wasserstein distance (TODO make that function public?).

        Returns
        -------
        The mean Wasserstein distance between batches of a distribution corresponding to one time point.
        """
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

    # TODO(@MUCDK) possibly offer two alternatives, once exact EMD with POT backend and once approximate,
    # faster with same solver as used for original problems
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

    @staticmethod
    def _get_interp_param(interpolation_parameter: Number, start: Number, intermediate: Number, end: Number) -> Number:
        if interpolation_parameter is not None and (0 > interpolation_parameter or interpolation_parameter > 1):
            raise ValueError("TODO: interpolation parameter must be in [0,1].")
        if start >= intermediate:
            raise ValueError("TODO: expected start < intermediate")
        if intermediate >= end:
            raise ValueError("TODO: expected intermediate < end")
        return (
            interpolation_parameter if interpolation_parameter is not None else (intermediate - start) / (end - start)
        )

    @property
    def temporal_key(self) -> Optional[str]:
        """Return temporal key."""
        return self._TEMPORAL_KEY

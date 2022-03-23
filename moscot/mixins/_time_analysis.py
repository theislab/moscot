from typing import Any, Dict, Literal, Optional
from numbers import Number
import logging

from sklearn.metrics.pairwise import pairwise_distances
import ot

from numpy import typing as npt
import numpy as np

from moscot.mixins._base_analysis import AnalysisMixin


class TemporalAnalysisMixin(AnalysisMixin):
    def validate_by_interpolation(
        self,
        start: Number,
        intermediate: Number,
        end: Number,
        interpolation_parameter: Optional[int] = None,
        valid_methods: Literal[
            "ot", "random", "random_with_growth", "source_to_intermediate", "intermediate_to_target"
        ] = ["ot", "random"],
        batch_key: Optional[str] = None,
        n_interpolated_cells: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Number]:
        """
        currently this assumes that we have preprocessed data which results in the questionable assumption that
        the held out data was also used for the preprocessing (whereas it should follow the independent preprocessing
        step of WOT
        """
        _validation_methods = {
            "ot", "random", "random_with_growth", "source_to_intermediate", "intermediate_to_target"
        }
        if not set(valid_methods).issubset(_validation_methods):
            raise ValueError(f"TODO: the only validation methods are {_validation_methods}")

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

        transport_matrix = self._compute_transport_map(start, end)
        gex_ot_old = self._interpolate_gex_with_ot_old(n_interpolated_cells, source_data, target_data, transport_matrix, interpolation_parameter)
        print(self._compute_wasserstein_distance(intermediate_data, gex_ot_old, **kwargs))

        result = {}
        if "ot" in valid_methods:
            gex_ot_interpolated = self._interpolate_gex_with_ot(
                n_interpolated_cells, source_data, target_data, start, end, interpolation_parameter
            )
            result["ot"] = self._compute_wasserstein_distance(intermediate_data, gex_ot_interpolated, **kwargs)

        if "random" in valid_methods:
            gex_randomly_interpolated = self._interpolate_gex_randomly(
                n_interpolated_cells, source_data, target_data, interpolation_parameter, **kwargs
            )
            result["random"] = self._compute_wasserstein_distance(
                intermediate_data, gex_randomly_interpolated, **kwargs
            )

        if "random_with_growth" in valid_methods:
            gex_randomly_interpolated_growth = self._interpolate_gex_randomly(
                len(intermediate_data),
                source_data,
                target_data,
                interpolation_parameter,
                growth_rates=growth_rates_source,
                **kwargs,
            )
            result["random_with_growth"] = self._compute_wasserstein_distance(
                intermediate_data, gex_randomly_interpolated_growth, **kwargs
            )

        if "source_to_intermediate" in valid_methods:
            result["source_to_intermediate"] = self._compute_wasserstein_distance(
                source_data, intermediate_data, **kwargs
            )

        if "intermediate_to_target" in valid_methods:
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
        numItermax: int = 1e6,
        **kwargs: Any,
    ) -> Number:
        cost_matrix = pairwise_distances(point_cloud_1, Y=point_cloud_2, metric="sqeuclidean", n_jobs=-1)
        a = [] if a is None else a
        b = [] if b is None else b
        return np.sqrt(ot.emd2(a, b, cost_matrix, numItermax=numItermax))
        #return np.sqrt(ot.emd2(a, b, cost_matrix, numItermax=numItermax, **kwargs)) #TODO: enable

    def _compute_transport_map(self, start: int, end: int, normalize: bool = False) -> npt.ArrayLike:
        if (start, end) not in self._problems.keys():
            steps = self._policy.plan(start=start, end=end)[start, end]
            transition_matrix = self._problems[steps[0]].solution.transport_matrix
            for i in range(len(steps) - 1):
                transition_matrix @= self._problems[steps[i + 1]].solution.transport_matrix
            if normalize:
                transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]
            return transition_matrix
        else:
            transition_matrix = self.solution[
                (start, end)
            ].solution.transport_matrix  # TODO(@MUCDK) report bug, one "solution" too much
            if normalize:
                transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]
                transition_matrix = np.nan_to_num(transition_matrix, nan=0.0)
            return transition_matrix

    def _interpolate_gex_with_ot_old(  # TODO: more efficient implementation
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        transport_matrix: npt.ArrayLike,
        interpolation_parameter: float = 0.5,
        adjust_by_growth=True,
    ) -> npt.ArrayLike:
        # TODO(@MUCDK): make online available
        # TODO(@MUCDK): check dimensions of arrays

        if adjust_by_growth:
            transport_matrix = transport_matrix / np.power(transport_matrix.sum(axis=0), 1.0 - interpolation_parameter)
            transport_matrix = np.nan_to_num(transport_matrix, nan=0)
        transport_matrix_flattened = transport_matrix.flatten(order="C").astype("float64")
        transport_matrix_flattened /= transport_matrix_flattened.sum() + 1e-3
        choices = np.random.choice(
            (len(source_data) * len(target_data)) + 1,
            p=np.concatenate(
                (transport_matrix_flattened, np.array([max(0, 1 - transport_matrix_flattened.sum())])), axis=0
            ),
            size=number_cells,
        )
        res = np.asarray(
            [
                source_data[i // len(target_data)] * (1 - interpolation_parameter)
                + target_data[i % len(target_data)] * interpolation_parameter
                for i in choices
                if i != (len(source_data) * len(target_data))
            ],
            dtype=np.float64,
        )

        n_to_replace = np.sum(choices == (len(source_data) * len(target_data)) + 1)
        rows_to_add = np.random.choice(
            len(res), replace=False, size=n_to_replace
        )  # this creates a slightly biased estimator but needs to be done due to numerical errors
        return np.concatenate((res, res[rows_to_add]), axis=0)

    def _interpolate_gex_with_ot(
        self,
        number_cells: int,
        source_data: npt.ArrayLike,
        target_data: npt.ArrayLike,
        start: Number,
        end: Number,
        interpolation_parameter: float = 0.5,
        account_for_unbalancedness: bool = True,
        batch_size: int = 64,
    ) -> npt.ArrayLike:

        rows_sampled, cols_sampled = self._sample_from_tmap(
            start,
            end,
            number_cells,
            len(source_data),
            len(target_data),
            batch_size,
            account_for_unbalancedness,
            interpolation_parameter,
        )
        print("len of rows_sampled are ", len(rows_sampled))
        print("cols sampled are ", cols_sampled)
        result = (
            source_data[np.repeat(rows_sampled, [len(col) for col in cols_sampled]), :] * (1 - interpolation_parameter)
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

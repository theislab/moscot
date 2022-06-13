from typing import Any, Dict, List, Tuple, Union, Generic, Optional, Sequence, TYPE_CHECKING, Mapping, Literal

from typing_extensions import Protocol
from scipy.sparse.linalg import LinearOperator

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_categorical_dtype

from anndata import AnnData

from moscot._types import ArrayLike, Numeric_t
from moscot.solvers._output import BaseSolverOutput
from moscot.problems._subset_policy import SubsetPolicy
from moscot.problems.base._compound_problem import B, K, ApplyOutput_t


class AnalysisMixinProtocol(Protocol[K, B]):
    """Protocol class."""

    adata: AnnData
    _policy: SubsetPolicy[K]
    solutions: Dict[Tuple[K, K], BaseSolverOutput]
    problems: Dict[Tuple[K, K], B]

    def _apply(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        start: Optional[K] = None,
        end: Optional[K] = None,
        forward: bool = True,
        return_all: bool = False,
        scale_by_marginals: bool = False,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        ...

    def _interpolate_transport(
        self: "AnalysisMixinProtocol[K, B]",
        path: Sequence[Tuple[K, K]],
        forward: bool = True,
        scale_by_marginals: bool = True,
    ) -> LinearOperator:
        ...

    def _flatten(self: "AnalysisMixinProtocol[K, B]", data: Dict[K, ArrayLike], *, key: Optional[str]) -> ArrayLike:
        ...


class AnalysisMixin(Generic[K, B]):
    """Base Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def cell_transition(
        self: AnalysisMixinProtocol[K, B],
        start: K,
        end: K,
        early_cells: Union[str, Mapping[str, Sequence[Any]]],
        late_cells: Union[str, Mapping[str, Sequence[Any]]],
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation: Literal["group", "cell"] = "group",
        statistic: Literal["mean", "top_k_mean"] = "mean",
        top_k: int = 5,
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
        forward
            If `True` computes the ancestors of the cells corresponding to time point `late`, else the descendants of
            the cells corresponding to time point `early`.
        aggregation:
            If `aggregation` is `group` the transition probabilities from the groups defined by `early_cells` are
            returned. If `aggregation` is `cell` the transition probablities for each cell are returned.
        statistic
            How to aggregate the distribution a cell is mapped onto. If `top_k_mean` only the `top_k` most likely ones
            are considered, all other probabilities are set to 0.
        top_k
            If `statistic` is `top_k_mean` ignore all matches of one cell in the other
            distributions which are not among the top k ones.

        Returns
        -------
        Transition matrix of groups between time points.
        """
        _split_mass = (statistic == "top_k_mean") or (aggregation == "cell")
        _early_cells_key, _early_cells = self._validate_args_cell_transition(early_cells)
        _late_cells_key, _late_cells = self._validate_args_cell_transition(late_cells)

        df_late = self.adata[self.adata.obs[self.temporal_key] == end].obs[[_late_cells_key]].copy()
        df_early = self.adata[self.adata.obs[self.temporal_key] == start].obs[[_early_cells_key]].copy()
        df_late["distribution"] = np.nan
        df_early["distribution"] = np.nan

        if aggregation == "group":
            transition_table = pd.DataFrame(
                np.zeros((len(_early_cells), len(_late_cells))), index=_early_cells, columns=_late_cells
            )
        elif aggregation == "cell":
            if forward:
                transition_table = pd.DataFrame(columns=_late_cells)
            else:
                transition_table = pd.DataFrame(index=_early_cells)
        else:
            raise NotImplementedError

        #result = self._cell_transition_helper(cells_to_iterate = _early_cells if forward else _late_cells, source_cells_df = df_early if forward else df_late, target_cells_df = df_late if forward else df_late, aggregation=aggregation, _source_cells_key=_early_cells_key if forward else _late_cells_key, forward=forward, split_mass=_split_mass, cell_dist_id=start if forward else end)


        #new
        
        #new_end


        if forward:
            _early_cells_present = set(_early_cells).intersection(set(df_early[_early_cells_key].unique()))
            for subset in _early_cells:
                result = self._cell_transition_helper(subset=subset, cells_present=_early_cells_present, source_cells_df = df_early, target_cells_df = df_late, aggregation=aggregation, _source_cells_key=_early_cells_key, forward=True, split_mass=split_mass, cell_dist_id=start)
                if result is None and aggregation == "group":
                        transition_table.loc[subset, :] = np.nan
                if statistic == "top_k_mean":
                    result = self._cell_transition_aggregation(result, statistic, top_k)

                if aggregation == "group":
                    df_late.loc[:, "distribution"] = result
                    target_cell_dist = (
                        df_late[df_late[_late_cells_key].isin(_late_cells)].groupby(_late_cells_key).sum()
                    )
                    target_cell_dist /= target_cell_dist.sum()
                    transition_table.loc[subset, :] = [
                        target_cell_dist.loc[cell_type, "distribution"]
                        if cell_type in target_cell_dist.distribution.index
                        else 0
                        for cell_type in _late_cells
                    ]
                elif aggregation == "cell":
                    current_early_cells = list(df_early[df_early[_early_cells_key] == subset].index)
                    df_late.loc[:, current_early_cells] = result
                    to_append = (
                        df_late[df_late[_late_cells_key].isin(_late_cells)].groupby(_late_cells_key).sum().transpose()
                    )
                    transition_table = transition_table.append(
                        to_append.drop(labels="distribution", axis=0), verify_integrity=True
                    )
                    df_late = df_late.drop(current_early_cells, axis=1)
                else:
                    raise NotImplementedError("TODO: aggregation must be `group` or `cell`.")
            return transition_table
        
        _late_cells_present = set(_late_cells).intersection(set(df_late[_late_cells_key].unique()))
        for subset in _late_cells:
            result = self._cell_transition_helper(subset=subset, cells_present=_late_cells_present, source_cells_df = df_late, target_cells_df = df_early, aggregation=aggregation, _source_cells_key=_late_cells_key, forward=False, split_mass=_split_mass, cell_dist_id=end)
            if result is None and aggregation == "group":
                transition_table.loc[:, subset] = np.nan
            
            if statistic == "top_k_mean":
                result = self._cell_transition_aggregation(result, statistic, top_k)

            if aggregation == "group":
                df_early.loc[:, "distribution"] = result
                filtered_df_early = df_early[df_early[_early_cells_key].isin(_early_cells)]

                target_cell_dist = filtered_df_early.groupby(_early_cells_key).sum()
                target_cell_dist /= target_cell_dist.sum()
                transition_table.loc[:, subset] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else 0
                    for cell_type in _early_cells
                ]
            elif aggregation == "cell":
                current_late_cells = list(df_late[df_late[_late_cells_key] == subset].index)
                df_early.loc[:, current_late_cells] = result
                to_append = df_early[df_early[_early_cells_key].isin(_early_cells)].groupby(_early_cells_key).sum()
                transition_table = pd.concat([transition_table.drop(labels="distribution", axis=1, errors="ignore"), to_append], axis=1)
                df_early = df_early.drop(current_late_cells, axis=1)
            else:
                raise NotImplementedError
        return transition_table

    def _cell_transition_helper(self, subset: str, cells_present: Sequence, source_cells_df: pd.DataFrame, target_cells_df: pd.DataFrame, aggregation: Union[Literal["group", "cell"]], _source_cells_key: str, forward: bool, split_mass: bool, cell_dist_id: K):
            
                if subset not in cells_present:
                    return None
                func = self.push if forward else self.pull
                try:
                    result = func(
                        start=start,
                        end=end,
                        data=_source_cells_key,
                        subset=subset,
                        normalize=True,
                        return_all=False,
                        scale_by_marginals=False,
                        split_mass=split_mass,
                    )
                except ValueError as e:
                    if "no mass" in str(e):  # TODO: adapt
                        logging.info(
                            f"No data points corresponding to {subset} found in `adata.obs[groups_key]` for {cell_dist_id}"
                        )
                        result = np.nan  # type: ignore[assignment]
                    else:
                        raise
                return result


    def _validate_args_cell_transition(
        self: AnalysisMixinProtocol[K, B], arg: Union[str, Mapping[str, Sequence[Any]]]
    ) -> Tuple[Union[str, Sequence[Any]], Sequence[Any]]:
        if isinstance(arg, str):
            if arg not in self.adata.obs:
                raise KeyError("TODO")
            if not is_categorical_dtype(self.adata.obs[arg]):
                raise TypeError(f"The column `{arg}` in `adata.obs` must be of categorical dtype.")
            if self.adata.obs[arg].isnull().values.any():  # TODO(giovp, MUCDK): why this check? remove?
                raise ValueError(f"The column `{arg}` in `adata.obs` contains NaN values. Please check.")
            return arg, self.adata.obs[arg].cat.categories
        if isinstance(arg, dict):
            if len(arg.keys()) > 1:
                raise ValueError(
                    f"The length of the dictionary is {len(arg)} but should be 1 as the data can only be filtered "
                    f"according to one column of `adata.obs`."
                )
            _key = list(arg.keys())[0]
            _val = arg[_key]
            if not is_categorical_dtype(self.adata.obs[_key]):
                raise TypeError(f"The column `{_key}` in `adata.obs` must be of categorical dtype.")
            if not set(_val).issubset(self.adata.obs[_key].cat.categories):
                raise ValueError(f"Not all values {_val} could be found in `adata.obs[{_key}]`.")
            if self.adata.obs[_key].isnull().values.any():
                raise ValueError(f"The column `{_key}` in `adata.obs` contains NaN values. Please check.")
            return _key, _val

    @staticmethod
    def _cell_transition_aggregation(
        result: np.ndarray, statistic: Literal["mean", "top_k_mean"] = "mean", top_k: int = 5
    ) -> np.ndarray:
        if statistic == "top_k_mean":
            if len(result) <= top_k:
                raise ValueError("TODO: `k` must be smaller than number of data points in distribution.")
            col_idx = np.array(range(result.shape[1]))
            low_col_k_indices = np.argpartition(result, -top_k, axis=0)[:-top_k, :]
            result[(low_col_k_indices.flatten(order="F"), np.repeat(col_idx, len(result) - top_k))] = 0
        else:
            raise NotImplementedError("TODO: not implemented.")
        return result.sum(axis=0)

    def _sample_from_tmap(
        self: AnalysisMixinProtocol[K, B],
        start: K,
        end: K,
        n_samples: int,
        source_dim: int,
        target_dim: int,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[Numeric_t] = None,
        seed: Optional[int] = None,
    ) -> Tuple[ArrayLike, List[ArrayLike]]:
        rng = np.random.RandomState(seed)
        if account_for_unbalancedness and interpolation_parameter is None:
            raise ValueError(
                "TODO: if unbalancedness is to be accounted for `interpolation_parameter` must be provided."
            )
        if interpolation_parameter is not None and (0 > interpolation_parameter or interpolation_parameter > 1):
            raise ValueError(f"TODO: interpolation parameter must be between 0 and 1 but is {interpolation_parameter}.")
        mass = np.ones(target_dim)
        if account_for_unbalancedness and interpolation_parameter is not None:
            col_sums = self._apply(
                start=start,
                end=end,
                normalize=True,
                forward=True,
                scale_by_marginals=False,
                explicit_steps=[(start, end)],
            )
            if TYPE_CHECKING:
                assert isinstance(col_sums, np.ndarray)
            col_sums = np.asarray(col_sums).squeeze() + 1e-12
            mass = mass / np.power(col_sums, 1 - interpolation_parameter)

        row_probability = np.asarray(
            self._apply(
                start=start,
                end=end,
                data=mass,
                normalize=True,
                forward=False,
                scale_by_marginals=False,
                explicit_steps=[(start, end)],
            )
        ).squeeze()

        rows_sampled = rng.choice(source_dim, p=row_probability / row_probability.sum(), size=n_samples)
        rows, counts = np.unique(rows_sampled, return_counts=True)
        all_cols_sampled = []
        for batch in range(0, len(rows), batch_size):
            rows_batch = rows[batch : batch + batch_size]
            counts_batch = counts[batch : batch + batch_size]
            data = np.zeros((source_dim, len(rows_batch)))
            data[rows_batch, range(len(rows_batch))] = 1

            col_p_given_row = np.asarray(
                self._apply(
                    start=start,
                    end=end,
                    data=data,
                    normalize=True,
                    forward=True,
                    scale_by_marginals=False,
                    explicit_steps=[(start, end)],
                )
            ).squeeze()
            if account_for_unbalancedness:
                if TYPE_CHECKING:
                    assert isinstance(col_sums, np.ndarray)
                col_p_given_row = col_p_given_row / col_sums[:, None]

            cols_sampled = [
                rng.choice(a=target_dim, size=counts_batch[i], p=col_p_given_row[:, i] / col_p_given_row[:, i].sum())
                for i in range(len(rows_batch))
            ]
            all_cols_sampled.extend(cols_sampled)
        return rows, all_cols_sampled

    def _interpolate_transport(
        self: AnalysisMixinProtocol[K, B],
        path: Sequence[
            Tuple[K, K]
        ],  # TODO(@giovp): rename this to 'explicit_steps', pass to policy.plan() and reintroduce (start, end) args
        forward: bool = True,
        scale_by_marginals: bool = True,
        **kwargs: Any,
    ) -> LinearOperator:
        """Interpolate transport matrix."""
        if TYPE_CHECKING:
            assert isinstance(self._policy, SubsetPolicy)
        # TODO(@MUCDK, @giovp, discuss what exactly this function should do, seems like it could be more generic)
        fst, *rest = path
        return self.solutions[fst].chain(
            [self.solutions[r] for r in rest], forward=forward, scale_by_marginals=scale_by_marginals
        )

    def _flatten(self: AnalysisMixinProtocol[K, B], data: Dict[K, ArrayLike], *, key: Optional[str]) -> ArrayLike:
        tmp = np.full(len(self.adata), np.nan)
        for k, v in data.items():
            mask = self.adata.obs[key] == k
            tmp[mask] = np.squeeze(v)
        return tmp

from typing import Any, Set, Dict, List, Tuple, Union, Generic, Literal, Mapping, Optional, Sequence, TYPE_CHECKING

from typing_extensions import Protocol
from scipy.sparse.linalg import LinearOperator
from pandas.core.dtypes.common import is_categorical_dtype
import pandas as pd

import numpy as np

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

    def _validate_args_cell_transition(
        self: "AnalysisMixinProtocol[K, B]", arg: Union[str, Mapping[str, Sequence[Any]]]
    ) -> Tuple[str, Sequence[Any]]:
        ...

    def _cell_transition_helper(
        self: "AnalysisMixinProtocol[K, B]",
        key_source: K,
        key_target: K,
        subset: str,
        cells_present: Set[Any],
        source_cells_key: str,
        forward: bool,
        split_mass: bool,
        cell_dist_id: K,
    ) -> Optional[ArrayLike]:
        ...


class AnalysisMixin(Generic[K, B]):
    """Base Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def cell_transition(
        self: AnalysisMixinProtocol[K, B],
        key: str,
        key_source: K,
        key_target: K,
        source_cells: Union[str, Mapping[str, Sequence[Any]]],
        target_cells: Union[str, Mapping[str, Sequence[Any]]],
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation: Literal["group", "cell"] = "group",
    ) -> pd.DataFrame:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        key
            Key according to which cells are allocated to distributions.
        key_source
            Time point corresponding to the early distribution.
        key_target
            Time point corresponding to the late distribution.
        source_cells
            Can be one of the following
                - if `source_cells` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in `anndata.AnnData.obs` ``['{source_cells}']``
                - if `source_cells` is of type `Mapping` its `key` should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its `value` to a subset of categories present in
                  `anndata.AnnData.obs` ``['{source_cells.keys()[0]}']``
        target_cells
            Can be one of the following
                - if `target_cells` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in `anndata.AnnData.obs` ``['{target_cells}']``
                - if `target_cells` is of type `Mapping` its `key` should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its `value` to a subset of categories present in
                  `anndata.AnnData.obs` ``['{target_cells.keys()[0]}']``
        forward
            If `True` computes transition from cells belonging to `source_cells` to cells belonging to `target_cells`.
        aggregation:
            If `aggregation` is `group` the transition probabilities from the groups defined by `source_cells` are
            returned. If `aggregation` is `cell` the transition probablities for each cell are returned.

        Returns
        -------
        Transition matrix of cells or groups of cells.
        """
        _split_mass = aggregation == "cell"
        _source_cells_key, _source_cells = self._validate_args_cell_transition(source_cells)
        _target_cells_key, _target_cells = self._validate_args_cell_transition(target_cells)

        df_target = self.adata[self.adata.obs[key] == key_target].obs[[_target_cells_key]].copy()
        df_source = self.adata[self.adata.obs[key] == key_source].obs[[_source_cells_key]].copy()

        if aggregation == "group":
            df_target["distribution"] = np.nan
            df_source["distribution"] = np.nan
            transition_table = pd.DataFrame(
                np.zeros((len(_source_cells), len(_target_cells))), index=_source_cells, columns=_target_cells
            )
        elif aggregation == "cell":
            if forward:
                transition_table = pd.DataFrame(columns=_target_cells)
            else:
                transition_table = pd.DataFrame(index=_source_cells)
        else:
            raise NotImplementedError

        error = NotImplementedError("TODO: aggregation must be `group` or `cell`.")
        if forward:
            _source_cells_present = set(_source_cells).intersection(set(df_source[_source_cells_key].cat.categories))
            for subset in _source_cells:
                result = self._cell_transition_helper(
                    key_source=key_source,
                    key_target=key_target,
                    subset=subset,
                    cells_present=_source_cells_present,
                    source_cells_key=_source_cells_key,
                    forward=True,
                    split_mass=_split_mass,
                    cell_dist_id=key_source,
                )
                if result is None:
                    continue

                if aggregation == "group":
                    df_target.loc[:, "distribution"] = result
                    target_cell_dist = (
                        df_target[df_target[_target_cells_key].isin(_target_cells)].groupby(_target_cells_key).sum()
                    )
                    target_cell_dist /= target_cell_dist.sum()
                    transition_table.loc[subset, :] = [
                        target_cell_dist.loc[cell_type, "distribution"]
                        if cell_type in target_cell_dist.distribution.index
                        else 0
                        for cell_type in _target_cells
                    ]
                elif aggregation == "cell":
                    current_source_cells = list(df_source[df_source[_source_cells_key] == subset].index)
                    df_target.loc[:, current_source_cells] = result
                    to_appkey_target = (
                        df_target[df_target[_target_cells_key].isin(_target_cells)]
                        .groupby(_target_cells_key)
                        .sum()
                        .transpose()
                    )
                    transition_table = pd.concat([transition_table, to_appkey_target], verify_integrity=True, axis=0)
                    df_target = df_target.drop(current_source_cells, axis=1)
                else:
                    raise error
            return transition_table

        _target_cells_present = set(_target_cells).intersection(set(df_target[_target_cells_key].cat.categories))
        for subset in _target_cells:
            result = self._cell_transition_helper(
                key_source=key_source,
                key_target=key_target,
                subset=subset,
                cells_present=_target_cells_present,
                source_cells_key=_target_cells_key,
                forward=False,
                split_mass=_split_mass,
                cell_dist_id=key_target,
            )

            if result is None:
                continue

            if aggregation == "group":
                df_source.loc[:, "distribution"] = result
                filtered_df_source = df_source[df_source[_source_cells_key].isin(_source_cells)]

                target_cell_dist = filtered_df_source.groupby(_source_cells_key).sum()
                target_cell_dist /= target_cell_dist.sum()
                transition_table.loc[:, subset] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else 0
                    for cell_type in _source_cells
                ]
            elif aggregation == "cell":
                current_target_cells = list(df_target[df_target[_target_cells_key] == subset].index)
                df_source.loc[:, current_target_cells] = result
                to_appkey_target = (
                    df_source[df_source[_source_cells_key].isin(_source_cells)].groupby(_source_cells_key).sum()
                )
                transition_table = pd.concat([transition_table, to_appkey_target], axis=1)
                df_source = df_source.drop(current_target_cells, axis=1)
            else:
                raise error
        return transition_table

    def _cell_transition_helper(
        self,
        key_source: K,
        key_target: K,
        subset: str,
        cells_present: Set[Any],
        source_cells_key: str,
        forward: bool,
        split_mass: bool,
        cell_dist_id: K,
    ) -> Optional[ArrayLike]:

        if subset not in cells_present:
            return None
        func = self.push if forward else self.pull  # type: ignore[attr-defined]
        try:
            result = func(
                start=key_source,
                end=key_target,
                data=source_cells_key,
                subset=subset,
                normalize=True,
                return_all=False,
                scale_by_marginals=False,
                split_mass=split_mass,
            )
        except ValueError as e:
            if "no mass" in str(e):  # TODO: adapt
                print(f"No data points corresponding to {subset} found in `adata.obs[groups_key]` for {cell_dist_id}")
                result = None
            else:
                raise
        return result

    def _validate_args_cell_transition(
        self: AnalysisMixinProtocol[K, B], arg: Union[str, Mapping[str, Sequence[Any]]]
    ) -> Tuple[str, Sequence[Any]]:
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

    def _sample_from_tmap(
        self: AnalysisMixinProtocol[K, B],
        key_source: K,
        key_target: K,
        n_samples: int,
        source_dim: int,
        target_dim: int,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[Numeric_t] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[Any], List[ArrayLike]]:
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
                start=key_source,
                end=key_target,
                normalize=True,
                forward=True,
                scale_by_marginals=False,
                explicit_steps=[(key_source, key_target)],
            )
            if TYPE_CHECKING:
                assert isinstance(col_sums, np.ndarray)
            col_sums = np.asarray(col_sums).squeeze() + 1e-12
            mass = mass / np.power(col_sums, 1 - interpolation_parameter)

        row_probability = np.asarray(
            self._apply(
                start=key_source,
                end=key_target,
                data=mass,
                normalize=True,
                forward=False,
                scale_by_marginals=False,
                explicit_steps=[(key_source, key_target)],
            )
        ).squeeze()

        rows_sampled = rng.choice(source_dim, p=row_probability / row_probability.sum(), size=n_samples)
        rows, counts = np.unique(rows_sampled, return_counts=True)
        all_cols_sampled: List[str] = []
        for batch in range(0, len(rows), batch_size):
            rows_batch = rows[batch : batch + batch_size]
            counts_batch = counts[batch : batch + batch_size]
            data = np.zeros((source_dim, len(rows_batch)))
            data[rows_batch, range(len(rows_batch))] = 1

            col_p_given_row = np.asarray(
                self._apply(
                    start=key_source,
                    end=key_target,
                    data=data,
                    normalize=True,
                    forward=True,
                    scale_by_marginals=False,
                    explicit_steps=[(key_source, key_target)],
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
        if TYPE_CHECKING:
            assert isinstance(rows, list)
            assert isinstance(all_cols_sampled, list)
        return rows, all_cols_sampled

    def _interpolate_transport(
        self: AnalysisMixinProtocol[K, B],
        path: Sequence[
            Tuple[K, K]
        ],  # TODO(@giovp): rename this to 'explicit_steps', pass to policy.plan() and reintroduce (key_source, key_target) args
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

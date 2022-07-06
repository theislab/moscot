from typing import Any, Set, Dict, List, Tuple, Union, Generic, Literal, Mapping, Optional, Sequence, TYPE_CHECKING

from typing_extensions import Protocol
from scipy.sparse.linalg import LinearOperator
import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike, Numeric_t
from moscot.solvers._output import BaseSolverOutput
from moscot._constants._constants import AggregationMode
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

    @staticmethod
    def _validate_args_cell_transition(
        adata: AnnData,
        arg: Optional[Union[str, Mapping[str, Sequence[Any]]]] = None,
    ) -> Tuple[Optional[str], Optional[Sequence[Any]]]:
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

    def _cell_transition_online(
        self: "AnalysisMixinProtocol[K, B]",
        key_source: K,
        key_target: K,
        key: Optional[str],
        source_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]],
        target_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]],
        forward: bool = False,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[AnnData] = None,
    ) -> pd.DataFrame:
        ...

    def _cell_transition_not_online(
        self: "AnalysisMixinProtocol[K, B]",
        key_source: K,
        key_target: K,
        key: Optional[str],
        source_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]],
        target_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]],
        forward: bool = False,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[AnnData] = None,
    ) -> pd.DataFrame:
        ...

    @staticmethod
    def _get_df_cell_transition(
        adata: AnnData,
        key: Optional[str],
        key_value: Optional[Any],
        annotation_key: Optional[str],
    ) -> pd.DataFrame:
        ...

    @staticmethod
    def _get_cell_indices(
        adata: AnnData,
        key: Optional[str],
        key_value: Optional[Any],
    ) -> pd.Index:
        ...

    @staticmethod
    def _get_categories_from_adata(
        adata: AnnData,
        key: Optional[str],
        key_value: Optional[Any],
        annotation_key: Optional[str],
    ) -> pd.Series:
        ...


class AnalysisMixin(Generic[K, B]):
    """Base Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _cell_transition_not_online(
        self: AnalysisMixinProtocol[K, B],
        key_source: K,
        key_target: K,
        key: Optional[str] = None,
        source_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]] = None,
        target_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]] = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[AnnData] = None,
    ) -> pd.DataFrame:

        source_annotation_key, source_annotations = self._validate_args_cell_transition(self.adata, source_cells)
        target_annotation_key, target_annotations = self._validate_args_cell_transition(
            self.adata if other_adata is None else other_adata, target_cells
        )

        df_source = self._get_df_cell_transition(self.adata, key, key_source, source_annotation_key)
        df_target = self._get_df_cell_transition(
            self.adata if other_adata is None else other_adata,
            key if other_adata is None else other_key,
            key_target,
            target_annotation_key,
        )

        source_cell_indices = self._get_cell_indices(self.adata, key, key_source)
        target_cell_indices = self._get_cell_indices(
            self.adata if other_adata is None else other_adata, key if other_adata is None else other_key, key_target
        )

        transition_matrix_indexed = pd.DataFrame(
            index=source_cell_indices,
            columns=target_cell_indices,
            data=np.array(self.solutions[key_source, key_target].transport_matrix),
        )
        aggregation_mode = AggregationMode(aggregation_mode)  # type: ignore[assignment]
        if forward:
            df_res = pd.DataFrame(index=source_cell_indices)
            if TYPE_CHECKING:
                assert target_annotations is not None  # this is checked in _cell_transition()
            for ct in target_annotations:  # TODO(@MUCKD) make more efficient?
                df_res[ct] = transition_matrix_indexed.loc[
                    :, df_target[df_target[target_annotation_key] == ct].index
                ].sum(axis=1)
            if aggregation_mode == "cell":
                return df_res.div(df_res.sum(axis=1), axis=0)
            df_res["source_cells_categories"] = self._get_categories_from_adata(
                self.adata,
                key,
                key_source,
                source_annotation_key,
            )
            unnormalized_tm = df_res.groupby("source_cells_categories").sum()
            normalized_tm = unnormalized_tm.div(unnormalized_tm.sum(axis=1), axis=0)
            return normalized_tm[normalized_tm.index.isin(source_annotations)]

        df_res = pd.DataFrame(index=source_annotations, columns=target_cell_indices)
        if TYPE_CHECKING:
            assert source_annotations is not None  # this is checked in _cell_transition()
        for ct in source_annotations:  # TODO(@MUCKD) make more efficient?
            df_res.loc[ct, :] = (
                transition_matrix_indexed.iloc[(df_source[source_annotation_key] == ct).values, :].sum(axis=0).squeeze()
            )
        if aggregation_mode == "cell":
            return df_res.div(df_res.sum(axis=0), axis=1)
        df_res.loc["target_cells_categories", :] = self._get_categories_from_adata(
            self.adata if other_adata is None else other_adata,
            key if other_adata is None else other_key,
            key_target,
            target_annotation_key,
        )
        unnormalized_tm = df_res.T.groupby("target_cells_categories").sum().T

        normalized_tm = unnormalized_tm.div(unnormalized_tm.sum(axis=0), axis=1)
        return normalized_tm[target_annotations]

    def _cell_transition(
        self: AnalysisMixinProtocol[K, B],
        key_source: K,
        key_target: K,
        key: Optional[str] = None,
        source_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]] = None,
        target_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]] = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        online: bool = False,
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        key_source
            Key identifying the source distribution.
        key_target
            Key identifying the target distribution.
        key
            Key according to which cells are allocated to distributions.
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
        aggregation_mode:
            If `aggregation_mode` is `group` the transition probabilities from the groups defined by `source_cells` are
            returned. If `aggregation_mode` is `cell` the transition probablities for each cell are returned.

        Returns
        -------
        Transition matrix of cells or groups of cells.
        """
        if (key_source is None or key_target is None) and other_adata is None:
            raise ValueError("TODO: distributions cannot be inferred from `adata` due to missing obs keys.")
        if (forward and target_cells is None) or (not forward and source_cells is None):
            raise ValueError("TODO: obs column according to which is grouped is required.")
        if online:
            return self._cell_transition_online(
                key_source, key_target, key, source_cells, target_cells, forward, aggregation_mode, other_key
            )
        return self._cell_transition_not_online(
            key_source, key_target, key, source_cells, target_cells, forward, aggregation_mode, other_key
        )

    def _cell_transition_online(
        self: AnalysisMixinProtocol[K, B],
        key_source: K,
        key_target: K,
        key: Optional[str],
        source_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]],
        target_cells: Optional[Union[str, Mapping[str, Sequence[Any]]]],
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        batch_size: Optional[int] = None,
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
    ) -> pd.DataFrame:
        aggregation_mode = AggregationMode(aggregation_mode)  # type: ignore[assignment]
        _split_mass = aggregation_mode == "cell"
        source_annotation_key, source_annotations = self._validate_args_cell_transition(self.adata, source_cells)
        target_annotation_key, target_annotations = self._validate_args_cell_transition(
            other_adata if other_adata is not None else self.adata, target_cells
        )

        df_source = self._get_df_cell_transition(self.adata, key, key_source, source_annotation_key)
        df_target = self._get_df_cell_transition(
            self.adata if other_adata is None else other_adata,
            key if other_adata is None else other_key,
            key_target,
            target_annotation_key,
        )

        _source_cells_present = set(source_annotations).intersection(
            set(df_source[source_annotation_key].cat.categories)
        )
        if not len(_source_cells_present):
            raise ValueError(f"TODO: None of {source_annotations} found in distribution corresponding to {key_source}.")
        _target_cells_present = set(target_annotations).intersection(
            set(df_target[target_annotation_key].cat.categories)
        )
        if not len(_target_cells_present):
            raise ValueError(f"TODO: None of {target_annotations} found in distribution corresponding to {key_target}.")

        error = NotImplementedError("TODO: aggregation_mode must be `group` or `cell`.")
        if aggregation_mode == AggregationMode.ANNOTATION:  # type: ignore[comparison-overlap]
            df_target["distribution"] = 0
            df_source["distribution"] = 0
            if forward:
                transition_table = pd.DataFrame(
                    np.zeros((len(_source_cells_present), len(target_annotations))),
                    index=_source_cells_present,
                    columns=target_annotations,
                )
            else:
                transition_table = pd.DataFrame(
                    np.zeros((len(source_annotations), len(_target_cells_present))),
                    index=source_annotations,
                    columns=_target_cells_present,
                )
        elif aggregation_mode == AggregationMode.CELL:  # type: ignore[comparison-overlap]
            transition_table = (
                pd.DataFrame(columns=target_annotations) if forward else pd.DataFrame(index=source_annotations)
            )
        else:
            raise error

        if forward:
            for subset in _source_cells_present:  # TODO(@MUCDK) introduce batch-wise application
                result = self._cell_transition_helper(
                    key_source=key_source,
                    key_target=key_target,
                    subset=subset,
                    cells_present=_source_cells_present,
                    source_cells_key=source_annotation_key,
                    forward=True,
                    split_mass=_split_mass,
                    cell_dist_id=key_source,
                )
                if aggregation_mode == AggregationMode.ANNOTATION:
                    df_target.loc[:, "distribution"] = result
                    target_cell_dist = (
                        df_target[df_target[target_annotation_key].isin(target_annotations)]
                        .groupby(target_annotation_key)
                        .sum()
                    )
                    target_cell_dist /= target_cell_dist.sum()
                    transition_table.loc[subset, :] = [
                        target_cell_dist.loc[cell_type, "distribution"]
                        if cell_type in target_cell_dist.distribution.index
                        else 0
                        for cell_type in target_annotations
                    ]
                elif aggregation_mode == AggregationMode.CELL:
                    current_source_cells = list(df_source[df_source[source_annotation_key] == subset].index)
                    df_target.loc[:, current_source_cells] = 0 if result is None else result
                    to_appkey_target = (
                        df_target[df_target[target_annotation_key].isin(target_annotations)]
                        .groupby(target_annotation_key)
                        .sum()
                        .transpose()
                    )
                    transition_table = pd.concat([transition_table, to_appkey_target], verify_integrity=True, axis=0)
                    df_target = df_target.drop(current_source_cells, axis=1)
                else:
                    raise error
            return transition_table.div(transition_table.sum(axis=1), axis=0)

        for subset in _target_cells_present:  # TODO(@MUCDK) introduce batch-wise application
            result = self._cell_transition_helper(
                key_source=key_source,
                key_target=key_target,
                subset=subset,
                cells_present=_target_cells_present,
                source_cells_key=target_annotation_key,
                forward=False,
                split_mass=_split_mass,
                cell_dist_id=key_target,
            )

            if aggregation_mode == AggregationMode.ANNOTATION:
                df_source.loc[:, "distribution"] = result
                filtered_df_source = df_source[df_source[source_annotation_key].isin(source_annotations)]

                target_cell_dist = filtered_df_source.groupby(source_annotation_key).sum()
                target_cell_dist /= target_cell_dist.sum()
                transition_table.loc[:, subset] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else 0
                    for cell_type in source_annotations
                ]
            elif aggregation_mode == AggregationMode.CELL:
                current_target_cells = list(df_target[df_target[target_annotation_key] == subset].index)
                df_source.loc[:, current_target_cells] = 0 if result is None else result
                to_appkey_target = (
                    df_source[df_source[source_annotation_key].isin(source_annotations)]
                    .groupby(source_annotation_key)
                    .sum()
                )
                transition_table = pd.concat([transition_table, to_appkey_target], axis=1)
                df_source = df_source.drop(current_target_cells, axis=1)
            else:
                raise error
        return transition_table.div(transition_table.sum(axis=0), axis=1)

    def _cell_transition_helper(
        self: AnalysisMixinProtocol[K, B],
        key_source: K,
        key_target: K,
        subset: str,
        cells_present: Set[Any],
        source_cells_key: str,
        forward: bool,
        split_mass: bool,
        cell_dist_id: K,
    ) -> ArrayLike:

        if subset not in cells_present:
            raise ValueError(f"TODO. Category {subset} not found")
        func = self.push if forward else self.pull  # type: ignore[attr-defined]
        return func(
            start=key_source,
            end=key_target,
            data=source_cells_key,
            subset=subset,
            normalize=True,
            return_all=False,
            scale_by_marginals=False,
            split_mass=split_mass,
        )

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
        return rows, all_cols_sampled  # type: ignore[return-value]

    def _interpolate_transport(
        self: AnalysisMixinProtocol[K, B],
        path: Sequence[
            Tuple[K, K]
        ],  # TODO(@giovp): rename this to 'explicit_steps', pass to policy.plan() and reintroduce (key_source, key_target) args
        forward: bool = True,
        scale_by_marginals: bool = True,
        **_: Any,
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

    @staticmethod
    def _validate_args_cell_transition(
        adata: AnnData,
        arg: Optional[Union[str, Mapping[str, Sequence[Any]]]],
    ) -> Tuple[Optional[str], Optional[Sequence[Any]]]:
        if arg is None:
            return None, None
        if isinstance(arg, str):
            if arg not in adata.obs:
                raise KeyError(f"TODO. {arg} not in adata.obs.columns")
            return arg, adata.obs[arg].cat.categories
        if isinstance(arg, dict):
            if len(arg) > 1:
                raise ValueError(f"Invalid dictionary length: `{len(arg)}` expected 1. ")
            _key, _val = next(iter(arg.items()))
            if not set(_val).issubset(adata.obs[_key].cat.categories):
                raise ValueError(f"Not all values {_val} could be found in `adata.obs[{_key}]`.")
            return _key, _val

    @staticmethod
    def _get_cell_indices(
        adata: AnnData,
        key: Optional[str] = None,
        key_value: Optional[Any] = None,
    ) -> pd.Index:
        if key is None:
            return adata.obs.index
        return adata[adata.obs[key] == key_value].obs.index

    @staticmethod
    def _get_categories_from_adata(
        adata: AnnData,
        key: Optional[str],
        key_value: Optional[Any],
        annotation_key: str,
    ) -> pd.Series:
        if key is None:
            return adata.obs[annotation_key]
        return adata[adata.obs[key] == key_value].obs[annotation_key]

    @staticmethod
    def _get_df_cell_transition(
        adata: AnnData,
        key: Optional[str],
        key_value: Optional[Any],
        annotation_key: Optional[str],
    ) -> pd.DataFrame:
        if key is None:
            return adata.obs.copy()
        return adata[adata.obs[key] == key_value].obs[[annotation_key]].copy()

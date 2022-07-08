from typing import Any, Dict, List, Tuple, Union, Generic, Literal, Iterable, Optional, Sequence, TYPE_CHECKING

from typing_extensions import Protocol
from scipy.sparse.linalg import LinearOperator
import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._types import Filter_t, ArrayLike, Numeric_t
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
        arg: Filter_t = None,
    ) -> Tuple[Optional[str], Optional[Iterable[Any]]]:
        ...

    def _cell_transition_online(
        self: "AnalysisMixinProtocol[K, B]",
        source_key: K,
        target_key: K,
        key: Optional[str],
        source_cells: Filter_t = None,
        target_cells: Filter_t = None,
        forward: bool = False,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[AnnData] = None,
    ) -> pd.DataFrame:
        ...

    def _cell_transition_not_online(
        self: "AnalysisMixinProtocol[K, B]",
        source_key: K,
        target_key: K,
        key: Optional[str],
        source_cells: Filter_t = None,
        target_cells: Filter_t = None,
        forward: bool = False,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[AnnData] = None,
    ) -> pd.DataFrame:
        ...

    @staticmethod
    def _get_df_cell_transition(
        adata: AnnData,
        key: Optional[str] = None,
        key_value: Optional[Any] = None,
        annotation_key: Optional[str] = None,
    ) -> pd.DataFrame:
        ...

    @staticmethod
    def _get_cell_indices(
        adata: AnnData,
        key: Optional[str] = None,
        key_value: Optional[Any] = None,
    ) -> pd.Index:
        ...

    @staticmethod
    def _get_categories_from_adata(
        adata: AnnData,
        key: Optional[str] = None,
        key_value: Optional[Any] = None,
        annotation_key: Optional[str] = None,
    ) -> pd.Series:
        ...

    @staticmethod
    def _check_argument_compatibility_cell_transition(
        key: Optional[str] = None,
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
        source_cells: Filter_t = None,
        target_cells: Filter_t = None,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        forward: bool = False,
    ) -> None:
        ...

    @staticmethod
    def _validate_annotations(
        df_source: pd.DataFrame,
        df_target: pd.DataFrame,
        source_annotation_key: Optional[str] = None,
        target_annotation_key: Optional[str] = None,
        source_annotations: Optional[Iterable[Any]] = None,
        target_annotations: Optional[Iterable[Any]] = None,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        forward: bool = False,
    ) -> Tuple[Iterable[Any], Iterable[Any]]:
        ...


class AnalysisMixin(Generic[K, B]):
    """Base Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _cell_transition(
        self: AnalysisMixinProtocol[K, B],
        source_key: K,
        target_key: K,
        key: Optional[str] = None,
        source_cells: Filter_t = None,
        target_cells: Filter_t = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        online: bool = False,
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
    ) -> pd.DataFrame:
        self._check_argument_compatibility_cell_transition(
            key=key,
            other_key=other_key,
            other_adata=other_adata,
            source_cells=source_cells,
            target_cells=target_cells,
            aggregation_mode=aggregation_mode,
            forward=forward,
        )
        if online:
            return self._cell_transition_online(
                source_key,
                target_key,
                key,
                source_cells,
                target_cells,
                forward,
                aggregation_mode,
                other_key,
                other_adata,
            )
        return self._cell_transition_not_online(
            source_key, target_key, key, source_cells, target_cells, forward, aggregation_mode, other_key, other_adata
        )

    def _cell_transition_not_online(
        self: AnalysisMixinProtocol[K, B],
        source_key: K,
        target_key: K,
        key: Optional[str] = None,
        source_cells: Filter_t = None,
        target_cells: Filter_t = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[AnnData] = None,
    ) -> pd.DataFrame:

        source_annotation_key, source_annotations = self._validate_args_cell_transition(self.adata, source_cells)
        target_annotation_key, target_annotations = self._validate_args_cell_transition(
            self.adata if other_adata is None else other_adata, target_cells
        )

        df_source = self._get_df_cell_transition(self.adata, key, source_key, source_annotation_key)
        df_target = self._get_df_cell_transition(
            self.adata if other_adata is None else other_adata,
            key if other_adata is None else other_key,
            target_key,
            target_annotation_key,
        )
        source_annotations_verified, target_annotations_verified = self._validate_annotations(
            df_source=df_source,
            df_target=df_target,
            source_annotation_key=source_annotation_key,
            target_annotation_key=target_annotation_key,
            source_annotations=source_annotations,
            target_annotations=target_annotations,
            aggregation_mode=aggregation_mode,
            forward=forward,
        )

        source_cell_indices = self._get_cell_indices(self.adata, key, source_key)
        target_cell_indices = self._get_cell_indices(
            self.adata if other_adata is None else other_adata, key if other_adata is None else other_key, target_key
        )

        transition_matrix_indexed = pd.DataFrame(
            index=source_cell_indices,
            columns=target_cell_indices,
            data=np.array(self.solutions[source_key, target_key].transport_matrix),
        )
        aggregation_mode = AggregationMode(aggregation_mode)  # type: ignore[assignment]
        if forward:
            df_res = pd.DataFrame(index=source_cell_indices)
            if TYPE_CHECKING:
                assert (
                    target_annotations is not None
                )  # this is checked in _check_argument_compatibility_cell_transition(
            for ct in target_annotations_verified:  # TODO(@MUCKD) make more efficient?
                df_res[ct] = transition_matrix_indexed.loc[
                    :, df_target[df_target[target_annotation_key] == ct].index
                ].sum(axis=1)
            if aggregation_mode == "cell":
                return df_res.div(df_res.sum(axis=1), axis=0)
            df_res["source_cells_categories"] = self._get_categories_from_adata(
                self.adata,
                key,
                source_key,
                source_annotation_key,
            )
            unnormalized_tm = df_res.groupby("source_cells_categories").sum()
            normalized_tm = unnormalized_tm.div(unnormalized_tm.sum(axis=1), axis=0)
            return normalized_tm[normalized_tm.index.isin(source_annotations_verified)]

        df_res = pd.DataFrame(index=source_annotations_verified, columns=target_cell_indices)
        if TYPE_CHECKING:
            assert (
                source_annotations_verified is not None
            )  # this is checked in _check_argument_compatibility_cell_transition(
        for ct in source_annotations_verified:  # TODO(@MUCKD) make more efficient?
            df_res.loc[ct, :] = (
                transition_matrix_indexed.iloc[(df_source[source_annotation_key] == ct).values, :].sum(axis=0).squeeze()
            )
        if aggregation_mode == "cell":
            return df_res.div(df_res.sum(axis=0), axis=1)
        df_res.loc["target_cells_categories", :] = self._get_categories_from_adata(
            self.adata if other_adata is None else other_adata,
            key if other_adata is None else other_key,
            target_key,
            target_annotation_key,
        )
        unnormalized_tm = df_res.T.groupby("target_cells_categories").sum().T

        normalized_tm = unnormalized_tm.div(unnormalized_tm.sum(axis=0), axis=1)
        return normalized_tm[target_annotations_verified]

    def _cell_transition_online(
        self: AnalysisMixinProtocol[K, B],
        source_key: K,
        target_key: K,
        key: Optional[str],
        source_cells: Filter_t = None,
        target_cells: Filter_t = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        aggregation_mode = AggregationMode(aggregation_mode)  # type: ignore[assignment]
        source_annotation_key, source_annotations = self._validate_args_cell_transition(self.adata, source_cells)
        target_annotation_key, target_annotations = self._validate_args_cell_transition(
            other_adata if other_adata is not None else self.adata, target_cells
        )

        df_source = self._get_df_cell_transition(self.adata, key, source_key, source_annotation_key)
        df_target = self._get_df_cell_transition(
            self.adata if other_adata is None else other_adata,
            key if other_adata is None else other_key,
            target_key,
            target_annotation_key,
        )

        source_annotations_verified, target_annotations_verified = self._validate_annotations(
            df_source=df_source,
            df_target=df_target,
            source_annotation_key=source_annotation_key,
            target_annotation_key=target_annotation_key,
            source_annotations=source_annotations,
            target_annotations=target_annotations,
            aggregation_mode=aggregation_mode,
            forward=forward,
        )

        if aggregation_mode == AggregationMode.ANNOTATION:  # type: ignore[comparison-overlap]
            df_target["distribution"] = 0
            df_source["distribution"] = 0
            if forward:
                transition_table = pd.DataFrame(
                    np.zeros((len(source_annotations_verified), len(target_annotations_verified))),
                    index=source_annotations_verified,
                    columns=target_annotations_verified,
                )
            else:
                transition_table = pd.DataFrame(
                    np.zeros((len(source_annotations_verified), len(target_annotations_verified))),
                    index=source_annotations_verified,
                    columns=target_annotations_verified,
                )
        elif aggregation_mode == AggregationMode.CELL:  # type: ignore[comparison-overlap]
            transition_table = (
                pd.DataFrame(columns=target_annotations_verified)
                if forward
                else pd.DataFrame(index=source_annotations_verified)
            )
        else:
            raise NotImplementedError("TODO: aggregation_mode must be `group` or `cell`.")

        if forward:
            if aggregation_mode == AggregationMode.ANNOTATION:
                for subset in source_annotations_verified:
                    result = self.push(  # TODO(@MUCDK) check how to make compatible with all policies
                        start=source_key,
                        end=target_key,
                        data=source_annotation_key,
                        subset=subset,
                        normalize=True,
                        return_all=False,
                        scale_by_marginals=False,
                        split_mass=False,
                    )

                    df_target.loc[:, "distribution"] = result
                    target_cell_dist = (
                        df_target[df_target[target_annotation_key].isin(target_annotations_verified)]
                        .groupby(target_annotation_key)
                        .sum()
                    )
                    target_cell_dist /= target_cell_dist.sum()
                    transition_table.loc[subset, :] = [
                        target_cell_dist.loc[cell_type, "distribution"]
                        if cell_type in target_cell_dist.distribution.index
                        else 0
                        for cell_type in target_annotations_verified
                    ]
            elif aggregation_mode == AggregationMode.CELL:
                if batch_size is None:
                    batch_size = len(df_source)
                for batch in range(0, len(df_source), batch_size):
                    result = self.push(  # TODO(@MUCDK) check how to make compatible with all policies
                        start=source_key,
                        end=target_key,
                        data=None,
                        subset=(batch, batch_size),
                        normalize=True,
                        return_all=False,
                        scale_by_marginals=False,
                        split_mass=True,
                    )
                    current_source_cells = list(
                        df_source.iloc[range(batch, min(batch + batch_size, len(df_source)))].index
                    )
                    df_target.loc[:, current_source_cells] = result
                    to_appkey_target = (
                        df_target[df_target[target_annotation_key].isin(target_annotations_verified)]
                        .groupby(target_annotation_key)
                        .sum()
                        .transpose()
                    )
                    transition_table = pd.concat([transition_table, to_appkey_target], verify_integrity=True, axis=0)
                    df_target = df_target.drop(current_source_cells, axis=1)
            else:
                NotImplementedError("TODO: aggregation_mode must be `group` or `cell`.")
            return transition_table.div(transition_table.sum(axis=1), axis=0)

        if aggregation_mode == AggregationMode.ANNOTATION:
            for subset in target_annotations_verified:
                result = self.pull(  # TODO(@MUCDK) check how to make compatible with all policies
                    start=source_key,
                    end=target_key,
                    data=target_annotation_key,
                    subset=subset,
                    normalize=True,
                    return_all=False,
                    scale_by_marginals=False,
                    split_mass=False,
                )

                df_source.loc[:, "distribution"] = result
                filtered_df_source = df_source[df_source[source_annotation_key].isin(source_annotations_verified)]

                target_cell_dist = filtered_df_source.groupby(source_annotation_key).sum()
                target_cell_dist /= target_cell_dist.sum()
                transition_table.loc[:, subset] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else 0
                    for cell_type in source_annotations_verified
                ]
        elif aggregation_mode == AggregationMode.CELL:
            if batch_size is None:
                batch_size = len(df_source)
            for batch in range(0, len(df_source), batch_size):
                result = self.pull(  # TODO(@MUCDK) check how to make compatible with all policies
                    start=source_key,
                    end=target_key,
                    data=None,
                    subset=(batch, batch_size),
                    normalize=True,
                    return_all=False,
                    scale_by_marginals=False,
                    split_mass=True,
                )

            current_target_cells = list(df_target.iloc[range(batch, min(batch + batch_size, len(df_source)))].index)
            df_source.loc[:, current_target_cells] = result
            to_appkey_target = (
                df_source[df_source[source_annotation_key].isin(source_annotations_verified)]
                .groupby(source_annotation_key)
                .sum()
            )
            transition_table = pd.concat([transition_table, to_appkey_target], axis=1)
            df_source = df_source.drop(current_target_cells, axis=1)
        else:
            raise NotImplementedError("TODO: aggregation_mode must be `group` or `cell`.")
        return transition_table.div(transition_table.sum(axis=0), axis=1)

    def _sample_from_tmap(
        self: AnalysisMixinProtocol[K, B],
        source_key: K,
        target_key: K,
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
                start=source_key,
                end=target_key,
                normalize=True,
                forward=True,
                scale_by_marginals=False,
                explicit_steps=[(source_key, target_key)],
            )
            if TYPE_CHECKING:
                assert isinstance(col_sums, np.ndarray)
            col_sums = np.asarray(col_sums).squeeze() + 1e-12
            mass = mass / np.power(col_sums, 1 - interpolation_parameter)

        row_probability = np.asarray(
            self._apply(
                start=source_key,
                end=target_key,
                data=mass,
                normalize=True,
                forward=False,
                scale_by_marginals=False,
                explicit_steps=[(source_key, target_key)],
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
                    start=source_key,
                    end=target_key,
                    data=data,
                    normalize=True,
                    forward=True,
                    scale_by_marginals=False,
                    explicit_steps=[(source_key, target_key)],
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
        # TODO(@giovp): rename this to 'explicit_steps', pass to policy.plan() and reintroduce (source_key, target_key) args
        path: Sequence[Tuple[K, K]],
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
        arg: Filter_t = None,
    ) -> Tuple[Optional[str], Optional[Iterable[Any]]]:
        if arg is None:
            return None, None
        if isinstance(arg, str):
            if arg not in adata.obs:
                raise KeyError(f"TODO. {arg} not in adata.obs.columns")
            return arg, adata.obs[arg].cat.categories
        if isinstance(arg, dict):
            if len(arg) > 1:
                raise ValueError(f"Invalid dictionary length: `{len(arg)}` expected 1. ")
            key, val = next(iter(arg.items()))
            if not set(val).issubset(adata.obs[key].cat.categories):
                raise ValueError(f"Not all values {val} could be found in `adata.obs[{key}]`.")
            return key, val

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
        key: Optional[str] = None,
        key_value: Optional[Any] = None,
        annotation_key: Optional[str] = None,
    ) -> pd.Series:
        if key is None:
            return adata.obs[annotation_key]
        return adata[adata.obs[key] == key_value].obs[annotation_key]

    @staticmethod
    def _get_df_cell_transition(
        adata: AnnData,
        key: Optional[str] = None,
        key_value: Optional[Any] = None,
        annotation_key: Optional[str] = None,
    ) -> pd.DataFrame:
        if key is None:
            return adata.obs.copy()
        return adata[adata.obs[key] == key_value].obs[[annotation_key]].copy()

    @staticmethod
    def _check_argument_compatibility_cell_transition(
        key: Optional[str] = None,
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
        source_cells: Filter_t = None,
        target_cells: Filter_t = None,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        forward: bool = False,
    ) -> None:
        if key is None and other_adata is None:
            raise ValueError("TODO: distributions cannot be inferred from `adata` due to missing obs keys.")
        if (forward and target_cells is None) or (not forward and source_cells is None):
            raise ValueError("TODO: obs column according to which is grouped is required.")
        if (AggregationMode(aggregation_mode) == AggregationMode.ANNOTATION) and (
            source_cells is None or target_cells is None
        ):
            raise ValueError("TODO: If `aggregation_mode` is `annotation` an `adata.obs` column must be provided.")

    @staticmethod
    def _validate_annotations(
        df_source: pd.DataFrame,
        df_target: pd.DataFrame,
        source_annotation_key: Optional[str] = None,
        target_annotation_key: Optional[str] = None,
        source_annotations: Optional[Iterable[Any]] = None,
        target_annotations: Optional[Iterable[Any]] = None,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        forward: bool = False,
    ) -> Tuple[Iterable[Any], Iterable[Any]]:
        if forward:
            if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition(
                assert target_annotations is not None
            target_annotations_verified = set(target_annotations).intersection(
                set(df_target[target_annotation_key].cat.categories)
            )
            if not len(target_annotations_verified):
                raise ValueError(
                    f"TODO: None of {target_annotations} found in distribution corresponding to {target_annotation_key}."
                )
            if aggregation_mode == AggregationMode.ANNOTATION:  # type: ignore[comparison-overlap]
                if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition()
                    assert source_annotations is not None
                source_annotations_verified = set(source_annotations).intersection(
                    set(df_source[source_annotation_key].cat.categories)
                )
            else:
                source_annotations_verified = [None]
            return source_annotations_verified, target_annotations_verified

        if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition(
            assert source_annotations is not None
        source_annotations_verified = set(source_annotations).intersection(  # type: ignore[assignment]
            set(df_source[source_annotation_key].cat.categories)
        )
        if not len(source_annotations_verified):
            raise ValueError(
                f"TODO: None of {source_annotations} found in distribution corresponding to {source_annotation_key}."
            )
        if aggregation_mode == AggregationMode.ANNOTATION:  # type: ignore[comparison-overlap]
            if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition()
                assert target_annotations is not None
            target_annotations_verified = set(target_annotations).intersection(
                set(df_target[target_annotation_key].cat.categories)
            )
        else:
            target_annotations_verified = [None]  # type: ignore[assignment]
        return source_annotations_verified, target_annotations_verified

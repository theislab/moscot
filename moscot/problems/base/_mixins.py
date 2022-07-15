from typing import Any, Dict, List, Tuple, Union, Generic, Literal, Iterable, Optional, Sequence, TYPE_CHECKING

from typing_extensions import Protocol
from scipy.sparse.linalg import LinearOperator
import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._types import Filter_t, ArrayLike, Numeric_t
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.base._utils import (
    _get_problem_key,
    _get_cell_indices,
    _get_df_cell_transition,
    _get_categories_from_adata,
    _validate_annotations_helper,
    _validate_args_cell_transition,
    _check_argument_compatibility_cell_transition,
)
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

    def push(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:
        ...

    def pull(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:
        ...

    def _cell_transition_online(
        self: "AnalysisMixinProtocol[K, B]",
        key: Optional[str],
        source_key: K,
        target_key: K,
        source_annotation: Filter_t = None,
        target_annotation: Filter_t = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        ...

    def _cell_transition_not_online(
        self: "AnalysisMixinProtocol[K, B]",
        key: Optional[str],
        source_key: K,
        target_key: K,
        source_annotation: Filter_t = None,
        target_annotation: Filter_t = None,
        forward: bool = False,
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[AnnData] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        ...

    def cell_aggregation_offline_helper(
        self: "AnalysisMixinProtocol[K, B]",
        adata: AnnData,
        key: Optional[str],
        df: pd.DataFrame,
        cell_indices_1: pd.Index,
        cell_indices_2: pd.Index,
        filter_key_1: K,
        annotation_key_1: Optional[str],
        filter_key_2: K,
        annotation_key_2: Optional[str],
        annotations: Iterable[Any],
        annotations_to_keep: Iterable[Any],
        aggregation_mode: Literal["annotation", "cell"],
        forward: bool,
    ) -> pd.DataFrame:
        ...

    def _validate_annotations(
        self: "AnalysisMixinProtocol[K, B]",
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

    def _cell_transition(  # TODO(@MUCDK) think about removing _cell_transition_non_online
        self: AnalysisMixinProtocol[K, B],
        *args: Any,
        online: bool,
        **kwargs: Any,
    ) -> pd.DataFrame:
        _check_argument_compatibility_cell_transition(*args, **kwargs)
        if online:
            return self._cell_transition_online(*args, **kwargs)
        return self._cell_transition_not_online(*args, **kwargs)

    def _cell_transition_not_online(
        self: AnalysisMixinProtocol[K, B],
        key: Optional[str],
        source_key: K,
        target_key: K,
        source_annotation: Filter_t = None,
        target_annotation: Filter_t = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[AnnData] = None,
        normalize: bool = True,
        **_: Any,
    ) -> pd.DataFrame:
        source_annotation_key, source_annotations = _validate_args_cell_transition(self.adata, source_annotation)
        target_annotation_key, target_annotations = _validate_args_cell_transition(
            self.adata if other_adata is None else other_adata, target_annotation
        )

        df_source = _get_df_cell_transition(
            self.adata,
            key,
            source_key,
            source_annotation_key,
        )
        df_target = _get_df_cell_transition(
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

        source_cell_indices = _get_cell_indices(self.adata, key, source_key)
        target_cell_indices = _get_cell_indices(
            self.adata if other_adata is None else other_adata, key if other_adata is None else other_key, target_key
        )

        problem_key = _get_problem_key(source_key, target_key)
        aggregation_mode = AggregationMode(aggregation_mode)  # type: ignore[assignment]

        if forward:
            transition_matrix = self.cell_aggregation_offline_helper(
                adata=self.adata,
                key=key,
                df=df_target,
                cell_indices_1=source_cell_indices,
                cell_indices_2=target_cell_indices,
                filter_key_1=source_key,
                annotation_key_1=source_annotation_key,
                filter_key_2=target_key,
                annotation_key_2=target_annotation_key,
                annotations=target_annotations_verified,
                annotations_to_keep=source_annotations_verified,
                aggregation_mode=aggregation_mode,
                forward=True,
            )
        else:
            transition_matrix = self.cell_aggregation_offline_helper(
                adata=self.adata if other_adata is None else other_adata,
                key=key if other_adata is None else other_key,
                df=df_source,
                cell_indices_1=target_cell_indices,
                cell_indices_2=source_cell_indices,
                filter_key_1=target_key,
                annotation_key_1=target_annotation_key,
                filter_key_2=source_key,
                annotation_key_2=source_annotation_key,
                annotations=source_annotations_verified,
                annotations_to_keep=target_annotations_verified,
                aggregation_mode=aggregation_mode,
                forward=False,
            )
        if normalize:
            transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

        return transition_matrix if forward else transition_matrix.T

    def _cell_transition_online(
        self: AnalysisMixinProtocol[K, B],
        key: Optional[str],
        source_key: K,
        target_key: K,
        source_annotation: Filter_t = None,
        target_annotation: Filter_t = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        **_: Any,
    ) -> pd.DataFrame:
        aggregation_mode = AggregationMode(aggregation_mode)  # type: ignore[assignment]
        source_annotation_key, source_annotations = _validate_args_cell_transition(
            self.adata, source_annotation
        )
        target_annotation_key, target_annotations = _validate_args_cell_transition(self.adata if other_adata is None else other_adata, target_annotation)

        df_source = _get_df_cell_transition(
            self.adata, key,
            source_key,
            source_annotation_key,
        )
        df_target = _get_df_cell_transition(
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
            transition_table = pd.DataFrame(
                np.zeros((len(source_annotations_verified), len(target_annotations_verified))),
                index=source_annotations_verified,
                columns=target_annotations_verified,
            )
            if forward:
                transition_table = self._annotation_aggregation_transition(
                    source_key=source_key,
                    target_key=target_key,
                    annotation_key=source_annotation_key,
                    annotations_1=source_annotations_verified,
                    annotations_2=target_annotations_verified,
                    df=df_target,
                    transition_table=transition_table,
                    forward=True,
                )
            else:
                transition_table = self._annotation_aggregation_transition(
                    source_key=source_key,
                    target_key=target_key,
                    annotation_key=target_annotation_key,
                    annotations_1=target_annotations_verified,
                    annotations_2=source_annotations_verified,
                    df=df_source,
                    transition_table=transition_table,
                    forward=False,
                )

        elif aggregation_mode == AggregationMode.CELL:  # type: ignore[comparison-overlap]
            transition_table = pd.DataFrame(
                columns=target_annotations_verified if forward else source_annotations_verified
            )
            if forward:
                transition_table = self._cell_aggregation_transition(
                    source_key=source_key,
                    target_key=target_key,
                    annotation_key=target_annotation_key,
                    annotations_1=source_annotations_verified,
                    annotations_2=target_annotations_verified,
                    df_1=df_target,
                    df_2=df_source,
                    transition_table=transition_table,
                    batch_size=batch_size,
                    forward=True,
                )
            else:
                transition_table = self._cell_aggregation_transition(
                    source_key=source_key,
                    target_key=target_key,
                    annotation_key=source_annotation_key,
                    annotations_1=target_annotations_verified,
                    annotations_2=source_annotations_verified,
                    df_1=df_source,
                    df_2=df_target,
                    transition_table=transition_table,
                    batch_size=batch_size,
                    forward=False,
                )

        else:
            raise NotImplementedError("TODO: aggregation_mode must be `group` or `cell`.")
        if normalize:
            transition_table = transition_table.div(transition_table.sum(axis=1), axis=0)
        return transition_table if forward else transition_table.T

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
        # TODO(@giovp): rename this to 'explicit_steps', pass to policy.plan() and reintroduce (source_key, target_key)
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

    def _validate_annotations(
        self: AnalysisMixinProtocol[K, B],
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
            source_annotations_verified = _validate_annotations_helper(
                df_source, source_annotation_key, source_annotations, aggregation_mode
            )
            return source_annotations_verified, target_annotations_verified

        if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition(
            assert source_annotations is not None
        source_annotations_verified = set(source_annotations).intersection(
            set(df_source[source_annotation_key].cat.categories)
        )
        if not len(source_annotations_verified):
            raise ValueError(
                f"TODO: None of {source_annotations} found in distribution corresponding to {source_annotation_key}."
            )
        target_annotations_verified = _validate_annotations_helper(  # type: ignore[assignment]
            df_target, target_annotation_key, target_annotations, aggregation_mode
        )
        return source_annotations_verified, target_annotations_verified

    def _annotation_aggregation_transition(
        self: AnalysisMixinProtocol[K, B],
        source_key: K,
        target_key: K,
        annotation_key: str,
        annotations_1: Iterable[Any],
        annotations_2: Iterable[Any],
        df: pd.DataFrame,
        transition_table: pd.DataFrame,
        forward: bool,
    ) -> pd.DataFrame:
        if not forward:
            transition_table = transition_table.T
        func = self.push if forward else self.pull
        for subset in annotations_1:
            result = func(  # TODO(@MUCDK) check how to make compatible with all policies
                start=source_key,
                end=target_key,
                data=annotation_key,
                subset=subset,
                normalize=True,
                return_all=False,
                scale_by_marginals=False,
                split_mass=False,
            )
            df["distribution"] = result
            cell_dist = df[df[annotation_key].isin(annotations_2)].groupby(annotation_key).sum()
            cell_dist /= cell_dist.sum()
            transition_table.loc[subset, :] = [
                cell_dist.loc[annotation, "distribution"] if annotation in cell_dist.distribution.index else 0
                for annotation in annotations_2
            ]
        return transition_table

    def _cell_aggregation_transition(
        self: AnalysisMixinProtocol[K, B],
        source_key: str,
        target_key: str,
        annotation_key: str,
        annotations_1: Iterable[Any],
        annotations_2: Iterable[Any],
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        transition_table: pd.DataFrame,
        batch_size: Optional[int],
        forward: bool,
    ) -> pd.DataFrame:
        func = self.push if forward else self.pull
        if batch_size is None:
            batch_size = len(df_2)
        for batch in range(0, len(df_2), batch_size):
            result = func(  # TODO(@MUCDK) check how to make compatible with all policies
                start=source_key,
                end=target_key,
                data=None,
                subset=(batch, batch_size),
                normalize=True,
                return_all=False,
                scale_by_marginals=False,
                split_mass=True,
            )
            current_cells = list(df_2.iloc[range(batch, min(batch + batch_size, len(df_2)))].index)
            df_1.loc[:, current_cells] = result
            to_app = df_1[df_1[annotation_key].isin(annotations_2)].groupby(annotation_key).sum().transpose()
            transition_table = pd.concat([transition_table, to_app], verify_integrity=True, axis=0)
            df_2 = df_2.drop(current_cells, axis=0)
        return transition_table

    def cell_aggregation_offline_helper(
        self: AnalysisMixinProtocol[K, B],
        adata: AnnData,
        key: Optional[str],
        df: pd.DataFrame,
        cell_indices_1: pd.Index,
        cell_indices_2: pd.Index,
        filter_key_1: K,
        annotation_key_1: Optional[str],
        filter_key_2: K,
        annotation_key_2: Optional[str],
        annotations: Iterable[Any],
        annotations_to_keep: Iterable[Any],
        aggregation_mode: Literal["annotation", "cell"],
        forward: bool,
    ) -> pd.DataFrame:
        key_added = "cell_annotations"  # TODO(giovp): use constants, expose.
        solution_key = _get_problem_key(
            filter_key_1 if forward else filter_key_2, filter_key_2 if forward else filter_key_1
        )
        tmap = np.array(self.solutions[solution_key].transport_matrix)
        transition_matrix_indexed = pd.DataFrame(
            index=cell_indices_1,
            columns=cell_indices_2,
            data=tmap if forward else tmap.T,
        )
        aggregation_mode = AggregationMode(aggregation_mode)  # type: ignore[assignment]
        df_res = pd.DataFrame(index=cell_indices_1)
        for annotation in annotations:
            df_res[annotation] = transition_matrix_indexed.loc[:, df[df[annotation_key_2] == annotation].index].sum(
                axis=1
            )
        if aggregation_mode == AggregationMode.CELL:  # type: ignore[comparison-overlap]
            return df_res
        df_res[key_added] = _get_categories_from_adata(
            adata=adata,
            key=key,
            key_value=filter_key_1,
            annotation_key=annotation_key_1,
        )
        transition_matrix = df_res.groupby(key_added).sum()
        return transition_matrix[transition_matrix.index.isin(annotations_to_keep)]

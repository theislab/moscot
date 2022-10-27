from typing import Any, Dict, List, Tuple, Union, Generic, Literal, Optional, Protocol, Sequence, TYPE_CHECKING

from scipy.sparse.linalg import LinearOperator
import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike, Numeric_t, Str_Dict_t
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.base._utils import (
    _get_df_cell_transition,
    _order_transition_matrix,
    _validate_annotations_helper,
    _validate_args_cell_transition,
    _check_argument_compatibility_cell_transition,
)
from moscot._constants._constants import Key, AdataKeys, PlottingKeys, AggregationMode, PlottingDefaults
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

    def _flatten(
        self: "AnalysisMixinProtocol[K, B]",
        data: Dict[K, ArrayLike],
        *,
        key: Optional[str],
    ) -> ArrayLike:
        ...

    def push(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:
        """Push distribution."""
        ...

    def pull(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:
        """Pull distribution."""
        ...

    def _cell_transition_online(
        self: "AnalysisMixinProtocol[K, B]",
        key: Optional[str],
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        ...

    def _validate_annotations(
        self: "AnalysisMixinProtocol[K, B]",
        df_source: pd.DataFrame,
        df_target: pd.DataFrame,
        source_annotation_key: Optional[str] = None,
        target_annotation_key: Optional[str] = None,
        source_annotations: Optional[List[Any]] = None,
        target_annotations: Optional[List[Any]] = None,
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        forward: bool = False,
    ) -> Tuple[List[Any], List[Any]]:
        ...


class AnalysisMixin(Generic[K, B]):
    """Base Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _cell_transition(
        self: AnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        key_added: Optional[str] = PlottingDefaults.CELL_TRANSITION,
        **kwargs: Any,
    ) -> pd.DataFrame:
        _check_argument_compatibility_cell_transition(
            source_annotation=source_groups,
            target_annotation=target_groups,
            **kwargs,
        )
        tm = self._cell_transition_online(
            source=source,
            target=target,
            source_groups=source_groups,
            target_groups=target_groups,
            **kwargs,
        )
        if key_added is not None:
            aggregation_mode = kwargs.pop("aggregation_mode")
            forward = kwargs.pop("forward")
            if aggregation_mode == AggregationMode.CELL and AggregationMode.CELL in self.adata.obs:
                raise KeyError(f"Aggregation is already present in `adata.obs[{aggregation_mode!r}]`.")
            plot_vars = {
                "transition_matrix": tm,
                "source": source,
                "target": target,
                "source_groups": source_groups
                if (not forward or aggregation_mode == AggregationMode.ANNOTATION)
                else AggregationMode.CELL,
                "target_groups": target_groups
                if (forward or aggregation_mode == AggregationMode.ANNOTATION)
                else AggregationMode.CELL,
            }
            Key.uns.set_plotting_vars(
                adata=self.adata,
                uns_key=AdataKeys.UNS,
                pl_func_key=PlottingKeys.CELL_TRANSITION,
                key=key_added,
                value=plot_vars,
            )
        return tm

    def _cell_transition_online(
        self: AnalysisMixinProtocol[K, B],
        key: Optional[str],
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        **_: Any,
    ) -> pd.DataFrame:
        aggregation_mode = AggregationMode(aggregation_mode)
        source_annotation_key, source_annotations, source_annotations_ordered = _validate_args_cell_transition(
            self.adata, source_groups
        )
        target_annotation_key, target_annotations, target_annotations_ordered = _validate_args_cell_transition(
            self.adata if other_adata is None else other_adata, target_groups
        )
        df_source = _get_df_cell_transition(
            self.adata,
            key,
            source,
            source_annotation_key,
        )
        df_target = _get_df_cell_transition(
            self.adata if other_adata is None else other_adata,
            key if other_adata is None else other_key,
            target,
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

        if aggregation_mode == AggregationMode.ANNOTATION:
            df_target["distribution"] = 0
            df_source["distribution"] = 0
            tm = pd.DataFrame(
                np.zeros((len(source_annotations_verified), len(target_annotations_verified))),
                index=source_annotations_verified,
                columns=target_annotations_verified,
            )
            if forward:
                tm = self._annotation_aggregation_transition(
                    source=source,
                    target=target,
                    annotation_key=source_annotation_key,
                    annotations_1=source_annotations_verified,
                    annotations_2=target_annotations_verified,
                    df=df_target,
                    tm=tm,
                    forward=True,
                )
            else:
                tm = self._annotation_aggregation_transition(
                    source=source,
                    target=target,
                    annotation_key=target_annotation_key,
                    annotations_1=target_annotations_verified,
                    annotations_2=source_annotations_verified,
                    df=df_source,
                    tm=tm,
                    forward=False,
                )
        elif aggregation_mode == AggregationMode.CELL:
            tm = pd.DataFrame(columns=target_annotations_verified if forward else source_annotations_verified)
            if forward:
                tm = self._cell_aggregation_transition(
                    source=source,
                    target=target,
                    annotation_key=target_annotation_key,
                    annotations_1=source_annotations_verified,
                    annotations_2=target_annotations_verified,
                    df_1=df_target,
                    df_2=df_source,
                    tm=tm,
                    batch_size=batch_size,
                    forward=True,
                )
            else:
                tm = self._cell_aggregation_transition(
                    source=source,
                    target=target,
                    annotation_key=source_annotation_key,
                    annotations_1=target_annotations_verified,
                    annotations_2=source_annotations_verified,
                    df_1=df_source,
                    df_2=df_target,
                    tm=tm,
                    batch_size=batch_size,
                    forward=False,
                )
        else:
            raise NotImplementedError(f"Aggregation mode `{aggregation_mode!r}` is not yet implemented.")

        if normalize:
            tm = tm.div(tm.sum(axis=1), axis=0)
        return _order_transition_matrix(
            tm=tm,
            source_annotations_verified=source_annotations_verified,
            target_annotations_verified=target_annotations_verified,
            source_annotations_ordered=source_annotations_ordered,
            target_annotations_ordered=target_annotations_ordered,
            forward=forward,
        )

    def _sample_from_tmap(
        self: AnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        n_samples: int,
        source_dim: int,
        target_dim: int,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[Numeric_t] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[Any], List[ArrayLike]]:
        rng = np.random.RandomState(seed)
        _ = "accounting"
        if account_for_unbalancedness and interpolation_parameter is None:
            raise ValueError("When accounting for unbalancedness, interpolation parameter must be provided.")
        if interpolation_parameter is not None and not (0 < interpolation_parameter < 1):
            raise ValueError(
                f"Expected interpolation parameter to be in interval `(0, 1)`, found `{interpolation_parameter}`."
            )
        mass = np.ones(target_dim)
        if account_for_unbalancedness and interpolation_parameter is not None:
            col_sums = self._apply(
                start=source,
                end=target,
                normalize=True,
                forward=True,
                scale_by_marginals=False,
                explicit_steps=[(source, target)],
            )
            if TYPE_CHECKING:
                assert isinstance(col_sums, np.ndarray)
            col_sums = np.asarray(col_sums).squeeze() + 1e-12
            mass = mass / np.power(col_sums, 1 - interpolation_parameter)

        row_probability = np.asarray(
            self._apply(
                start=source,
                end=target,
                data=mass,
                normalize=True,
                forward=False,
                scale_by_marginals=False,
                explicit_steps=[(source, target)],
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
                    start=source,
                    end=target,
                    data=data,
                    normalize=True,
                    forward=True,
                    scale_by_marginals=False,
                    explicit_steps=[(source, target)],
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
        source_annotations: Optional[List[Any]] = None,
        target_annotations: Optional[List[Any]] = None,
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        forward: bool = False,
    ) -> Tuple[List[Any], List[Any]]:
        if forward:
            if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition(
                assert target_annotations is not None
            target_annotations_verified = list(
                set(df_target[target_annotation_key].cat.categories).intersection(target_annotations)
            )
            if not len(target_annotations_verified):
                raise ValueError(f"None of `{target_annotations}`, found in the target annotations.")
            source_annotations_verified = _validate_annotations_helper(
                df_source, source_annotation_key, source_annotations, aggregation_mode
            )
            return source_annotations_verified, target_annotations_verified

        if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition(
            assert source_annotations is not None
        source_annotations_verified = list(
            set(df_source[source_annotation_key].cat.categories).intersection(set(source_annotations))
        )
        if not len(source_annotations_verified):
            raise ValueError(f"None of `{source_annotations}`, found in the source annotations.")
        target_annotations_verified = _validate_annotations_helper(
            df_target, target_annotation_key, target_annotations, aggregation_mode
        )
        return source_annotations_verified, target_annotations_verified

    def _annotation_aggregation_transition(
        self: AnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        annotation_key: str,
        annotations_1: List[Any],
        annotations_2: List[Any],
        df: pd.DataFrame,
        tm: pd.DataFrame,
        forward: bool,
    ) -> pd.DataFrame:
        if not forward:
            tm = tm.T
        func = self.push if forward else self.pull
        for subset in annotations_1:
            result = func(  # TODO(@MUCDK) check how to make compatible with all policies
                start=source,
                end=target,
                data=annotation_key,
                subset=subset,
                normalize=True,
                return_all=False,
                scale_by_marginals=False,
                split_mass=False,
                key_added=None,
                return_data=True,
            )
            df["distribution"] = result
            cell_dist = df[df[annotation_key].isin(annotations_2)].groupby(annotation_key).sum()
            cell_dist /= cell_dist.sum()
            tm.loc[subset, :] = [
                cell_dist.loc[annotation, "distribution"] if annotation in cell_dist.distribution.index else 0
                for annotation in annotations_2
            ]
        return tm

    def _cell_aggregation_transition(
        self: AnalysisMixinProtocol[K, B],
        source: str,
        target: str,
        annotation_key: str,
        # TODO(MUCDK): unused variables, del below
        annotations_1: List[Any],
        annotations_2: List[Any],
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        tm: pd.DataFrame,
        batch_size: Optional[int],
        forward: bool,
    ) -> pd.DataFrame:
        func = self.push if forward else self.pull
        if batch_size is None:
            batch_size = len(df_2)
        for batch in range(0, len(df_2), batch_size):
            result = func(  # TODO(@MUCDK) check how to make compatible with all policies
                start=source,
                end=target,
                data=None,
                subset=(batch, batch_size),
                normalize=True,
                return_all=False,
                scale_by_marginals=False,
                split_mass=True,
                key_added=None,
                return_data=True,
            )
            current_cells = df_2.iloc[range(batch, min(batch + batch_size, len(df_2)))].index.tolist()
            df_1.loc[:, current_cells] = result
            to_app = df_1[df_1[annotation_key].isin(annotations_2)].groupby(annotation_key).sum().transpose()
            tm = pd.concat([tm, to_app], verify_integrity=True, axis=0)
            df_1 = df_1.drop(current_cells, axis=1)
        return tm

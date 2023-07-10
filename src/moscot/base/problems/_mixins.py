from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator

import scanpy as sc
from anndata import AnnData

from moscot import _constants
from moscot._types import ArrayLike, Numeric_t, Str_Dict_t
from moscot.base.output import BaseSolverOutput
from moscot.base.problems._utils import (
    _check_argument_compatibility_cell_transition,
    _correlation_test,
    _get_df_cell_transition,
    _order_transition_matrix,
    _validate_annotations,
    _validate_args_cell_transition,
)
from moscot.base.problems.compound_problem import ApplyOutput_t, B, K
from moscot.plotting._utils import set_plotting_vars
from moscot.utils.data import transcription_factors
from moscot.utils.subset_policy import SubsetPolicy

__all__ = ["AnalysisMixin"]


class AnalysisMixinProtocol(Protocol[K, B]):
    """Protocol class."""

    adata: AnnData
    _policy: SubsetPolicy[K]
    solutions: Dict[Tuple[K, K], BaseSolverOutput]
    problems: Dict[Tuple[K, K], B]

    def _apply(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        source: Optional[K] = None,
        target: Optional[K] = None,
        forward: bool = True,
        return_all: bool = False,
        scale_by_marginals: bool = False,
        **kwargs: Any,
    ) -> ApplyOutput_t[K]:
        ...

    def _interpolate_transport(
        self: "AnalysisMixinProtocol[K, B]",
        path: Sequence[Tuple[K, K]],
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
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        key_added: Optional[str] = _constants.CELL_TRANSITION,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if aggregation_mode == "annotation" and (source_groups is None or target_groups is None):
            raise ValueError(
                "If `aggregation_mode='annotation'`, `source_groups` and `target_groups` cannot be `None`."
            )
        if aggregation_mode == "cell" and source_groups is None and target_groups is None:
            raise ValueError("At least one of `source_groups` and `target_group` must be specified.")

        _check_argument_compatibility_cell_transition(
            source_annotation=source_groups,
            target_annotation=target_groups,
            aggregation_mode=aggregation_mode,
            **kwargs,
        )
        tm = self._cell_transition_online(
            source=source,
            target=target,
            source_groups=source_groups,
            target_groups=target_groups,
            aggregation_mode=aggregation_mode,
            **kwargs,
        )
        if key_added is not None:
            forward = kwargs.pop("forward")
            if aggregation_mode == "cell" and "cell" in self.adata.obs:
                raise KeyError(f"Aggregation is already present in `adata.obs[{aggregation_mode!r}]`.")
            plot_vars = {
                "source": source,
                "target": target,
                "source_groups": source_groups if (not forward or aggregation_mode == "annotation") else "cell",
                "target_groups": target_groups if (forward or aggregation_mode == "annotation") else "cell",
                "transition_matrix": tm,
            }
            set_plotting_vars(
                self.adata,
                _constants.CELL_TRANSITION,
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
        source_annotation_key, source_annotations, source_annotations_ordered = _validate_args_cell_transition(
            self.adata, source_groups
        )
        target_annotation_key, target_annotations, target_annotations_ordered = _validate_args_cell_transition(
            self.adata if other_adata is None else other_adata, target_groups
        )
        df_source = _get_df_cell_transition(
            self.adata,
            [source_annotation_key, target_annotation_key],
            key,
            source,
        )
        df_target = _get_df_cell_transition(
            self.adata if other_adata is None else other_adata,
            [source_annotation_key, target_annotation_key],
            key if other_adata is None else other_key,
            target,
        )

        source_annotations_verified, target_annotations_verified = _validate_annotations(
            df_source=df_source,
            df_target=df_target,
            source_annotation_key=source_annotation_key,
            target_annotation_key=target_annotation_key,
            source_annotations=source_annotations,
            target_annotations=target_annotations,
            aggregation_mode=aggregation_mode,
            forward=forward,
        )

        if aggregation_mode == "annotation":
            df_target["distribution"] = 0
            df_source["distribution"] = 0
            tm = pd.DataFrame(
                np.zeros((len(source_annotations_verified), len(target_annotations_verified))),
                index=source_annotations_verified,
                columns=target_annotations_verified,
            )
            if forward:
                tm = self._annotation_aggregation_transition(  # type: ignore[attr-defined]
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
                tm = self._annotation_aggregation_transition(  # type: ignore[attr-defined]
                    source=source,
                    target=target,
                    annotation_key=target_annotation_key,
                    annotations_1=target_annotations_verified,
                    annotations_2=source_annotations_verified,
                    df=df_source,
                    tm=tm,
                    forward=False,
                )
        elif aggregation_mode == "cell":
            tm = pd.DataFrame(columns=target_annotations_verified if forward else source_annotations_verified)
            if forward:
                tm = self._cell_aggregation_transition(  # type: ignore[attr-defined]
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
                tm = self._cell_aggregation_transition(  # type: ignore[attr-defined]
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
        if account_for_unbalancedness and interpolation_parameter is None:
            raise ValueError("When accounting for unbalancedness, interpolation parameter must be provided.")
        if interpolation_parameter is not None and not (0 < interpolation_parameter < 1):
            raise ValueError(
                f"Expected interpolation parameter to be in interval `(0, 1)`, found `{interpolation_parameter}`."
            )
        mass = np.ones(target_dim)
        if account_for_unbalancedness and interpolation_parameter is not None:
            col_sums = self._apply(
                source=source,
                target=target,
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
                source=source,
                target=target,
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
                    source=source,
                    target=target,
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
        scale_by_marginals: bool = True,
        **_: Any,
    ) -> LinearOperator:
        """Interpolate transport matrix."""
        if TYPE_CHECKING:
            assert isinstance(self._policy, SubsetPolicy)
        # TODO(@MUCDK, @giovp, discuss what exactly this function should do, seems like it could be more generic)
        fst, *rest = path
        return self.solutions[fst].chain([self.solutions[r] for r in rest], scale_by_marginals=scale_by_marginals)

    def _flatten(self: AnalysisMixinProtocol[K, B], data: Dict[K, ArrayLike], *, key: Optional[str]) -> ArrayLike:
        tmp = np.full(len(self.adata), np.nan)
        for k, v in data.items():
            mask = self.adata.obs[key] == k
            tmp[mask] = np.squeeze(v)
        return tmp

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
                source=source,
                target=target,
                data=annotation_key,
                subset=subset,
                normalize=True,
                return_all=False,
                scale_by_marginals=False,
                split_mass=False,
                key_added=None,
            )
            df["distribution"] = result
            cell_dist = df[df[annotation_key].isin(annotations_2)].groupby(annotation_key).sum(numeric_only=True)
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
                source=source,
                target=target,
                data=None,
                subset=(batch, batch_size),
                normalize=True,
                return_all=False,
                scale_by_marginals=False,
                split_mass=True,
                key_added=None,
            )
            current_cells = df_2.iloc[range(batch, min(batch + batch_size, len(df_2)))].index.tolist()
            df_1.loc[:, current_cells] = result
            to_app = df_1[df_1[annotation_key].isin(annotations_2)].groupby(annotation_key).sum().transpose()
            tm = pd.concat([tm, to_app], verify_integrity=True, axis=0)
            df_1 = df_1.drop(current_cells, axis=1)
        return tm

    # adapted from:
    # https://github.com/theislab/cellrank/blob/master/cellrank/_utils/_utils.py#L392
    def compute_feature_correlation(
        self: AnalysisMixinProtocol[K, B],
        obs_key: str,
        corr_method: Literal["pearson", "spearman"] = "pearson",
        significance_method: Literal["fisher", "perm_test"] = "fisher",
        annotation: Optional[Dict[str, Iterable[str]]] = None,
        layer: Optional[str] = None,
        features: Optional[Union[List[str], Literal["human", "mouse", "drosophila"]]] = None,
        confidence_level: float = 0.95,
        n_perms: int = 1000,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute correlation of push-forward or pull-back distribution with features.

        Correlates a feature, e.g., counts of a gene, with probabilities of cells mapped to a set of cells such as
        the push-forward or pull-back distributions.

        .. seealso::
            - TODO: create and link an example

        Parameters
        ----------
        obs_key
            Key in :attr:`~anndata.AnnData.obs` containing the push-forward or pull-back distribution.
        corr_method
            Which type of correlation to compute, either ``'pearson'`` or ``'spearman'``.
        significance_method
            Mode to use when calculating p-values and confidence intervals. Valid options are:

            - ``'fisher'`` - Fisher transformation :cite:`fisher:21`.
            - ``'perm_test'`` - `permutation test <https://en.wikipedia.org/wiki/Permutation_test>`_.
        annotation
            How to subset the data when computing the correlation:

            - :obj:`None` - do not subset the data.
            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where categorical data is stored.
            - :class:`dict` - a dictionary with one key corresponding to a categorical column in
              :attr:`~anndata.AnnData.obs` and values to a subset of categories.
        layer
            Key in :attr:`~anndata.AnnData.layers` from which to get the expression.
            If :obj:`None`, use :attr:`~anndata.AnnData.X`.
        features
            Features in :class:`~anndata.AnnData` to correlate with
            :attr:`obs['{obs_key}'] <anndata.AnnData.obs>`:

            - :obj:`None` - all features from :attr:`~anndata.AnnData.var` will be taken into account.
            - :obj:`list` - subset of :attr:`~anndata.AnnData.var_names` or :attr:`~anndata.AnnData.obs_names`.
            - ``'human'``, ``'mouse'``, or ``'drosophila'`` - the features are subsetted to the transcription factors
              from :func:`~moscot.utils.data.transcription_factors`.
        confidence_level
            Confidence level for the confidence interval calculation. Must be in interval :math:`[0, 1]`.
        n_perms
            Number of permutations to use when ``method = 'perm_test'``.
        seed
            Random seed when ``method = 'perm_test'``.
        kwargs
            Keyword arguments for parallelization, e.g., ``n_jobs``.

        Returns
        -------
        Dataframe of shape ``(n_features, 5)`` containing the following columns, one for each feature:

        - ``'corr'`` - correlation between the count data and push/pull distributions.
        - ``'pval'`` - calculated p-values for double-sided test.
        - ``'qval'`` - corrected p-values using the `Benjamini-Hochberg
          <https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure>`_ method
          at :math:`0.05` level.
        - ``'ci_low'`` - lower bound of the ``confidence_level`` correlation confidence interval.
        - ``'ci_high'`` - upper bound of the ``confidence_level`` correlation confidence interval.
        """
        if obs_key not in self.adata.obs:
            raise KeyError(f"Unable to access data in `adata.obs[{obs_key!r}]`.")

        if annotation is not None:
            annotation_key, annotation_vals = next(iter(annotation.items()))
            if annotation_key not in self.adata.obs:
                raise KeyError(f"Unable to access data in `adata.obs[{annotation_key!r}]`.")
            if not isinstance(annotation_vals, Iterable):
                raise TypeError("`annotation` expected to be dictionary of length 1 with value being an iterable.")
            adata = self.adata[self.adata.obs[annotation_key].isin(annotation_vals)]
        else:
            adata = self.adata

        adata = adata[~adata.obs[obs_key].isnull()]
        if not adata.n_obs:
            raise ValueError(f"`adata.obs[{obs_key!r}]` only contains NaN values.")
        distribution: pd.DataFrame = adata.obs[[obs_key]]

        if isinstance(features, str):
            tfs = transcription_factors(organism=features)
            features = list(set(tfs).intersection(adata.var_names))
            if not features:
                raise KeyError("No common transcription factors found in the data base.")
        elif features is None:
            features = list(self.adata.var_names)

        return _correlation_test(
            X=sc.get.obs_df(adata, keys=features, layer=layer).values,
            Y=distribution,
            feature_names=features,
            corr_method=corr_method,
            significance_method=significance_method,
            confidence_level=confidence_level,
            n_perms=n_perms,
            seed=seed,
            **kwargs,
        )

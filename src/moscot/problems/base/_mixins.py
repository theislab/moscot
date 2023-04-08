from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Generic,
    Literal,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
)

from scipy.sparse.linalg import LinearOperator
import pandas as pd

import numpy as np

from anndata import AnnData
import scanpy as sc

from moscot._types import ArrayLike, Numeric_t, Str_Dict_t
from moscot._logging import logger
from moscot._docs._docs import d
from moscot.utils._data import TranscriptionFactors
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.base._utils import (
    _correlation_test,
    _validate_annotations,
    _get_df_cell_transition,
    _order_transition_matrix,
    _validate_args_cell_transition,
    _check_argument_compatibility_cell_transition,
)
from moscot._constants._constants import (
    Key,
    CorrMethod,
    PlottingKeys,
    CorrTestMethod,
    AggregationMode,
    PlottingDefaults,
)
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
                source=source,
                target=target,
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

    # adapted from CellRank (github.com/theislab/cellrank)
    @d.dedent
    def compute_feature_correlation(
        self: AnalysisMixinProtocol[K, B],
        obs_key: str,
        corr_method: CorrMethod = CorrMethod.PEARSON,
        significance_method: Literal["fischer", "perm_test"] = CorrTestMethod.FISCHER,
        annotation: Optional[Dict[str, Iterable[str]]] = None,
        layer: Optional[str] = None,
        features: Optional[Union[List[str], Literal["human", "mouse", "drosophila"]]] = None,
        confidence_level: float = 0.95,
        n_perms: int = 1000,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute correlation of push or pull distribution with features.

        Correlates a feature (e.g. counts of a gene) with probabilities of cells mapped to a set of cells, e.g.
        a pull back or push forward distribution.

        Parameters
        ----------
        obs_key
            Column of :attr:`anndata.AnnData.obs` containing push-forward or pull-back distributions.
        corr_method
            Which type of correlation to compute, options are `pearson`, and `spearman`.
        significance_method
            Mode to use when calculating p-values and confidence intervals. Valid options are:

                - `fischer` - use Fischer transformation :cite:`fischer:21`.
                - `perm_test` - use permutation test.

        annotation
            If not `None`, this defines the subset of data to be considered when computing the correlation.
            Its key should correspond to a key in
            :attr:`anndata.AnnData.obs` and its value to an iterable containing a subset of categories present in
            :attr:`anndata.AnnData.obs` ``['{annotation.keys()[0]}']``.
        layer
            Key from :attr:`anndata.AnnData.layers` from which to get the expression.
            If `None`, use :attr:`anndata.AnnData.X`.
        features
            Features in :class:`anndata.AnnData` which the correlation
            of ``anndata.AnnData.obs['{obs_key}']`` is computed with:

                - If `None`, all features from :attr:`anndata.AnnData.var` will be taken into account.
                - If of type :obj:`list`, the elements should be from :attr:`anndata.AnnData.var_names` or
                  :attr:`anndata.AnnData.obs_names`.
                - If `human`, `mouse`, or `drosophila`, the features are subsetted to transcription factors,
                  see :class:`moscot.utils._data.TranscriptionFactors`.

        confidence_level
            Confidence level for the confidence interval calculation. Must be in interval `[0, 1]`.
        n_perms
            Number of permutations to use when ``method = perm_test``.
        seed
            Random seed when ``method = perm_test``.
        kwargs
            Keyword arguments for :func:`moscot._utils.parallelize`, e.g. `n_jobs`.

        Returns
        -------
        Dataframe of shape ``(n_features, 5)`` containing the following columns, one for each lineage:

            - `corr` - correlation between the count data and push/pull distributions.
            - `pval` - calculated p-values for double-sided test.
            - `qval` - corrected p-values using Benjamini-Hochberg method at level `0.05`.
            - `ci_low` - lower bound of the ``confidence_level`` correlation confidence interval.
            - `ci_high` - upper bound of the ``confidence_level`` correlation confidence interval.

        """
        if obs_key not in self.adata.obs:
            raise KeyError(f"Unable to access data in `adata.obs[{obs_key!r}]`.")

        significance_method = CorrTestMethod(significance_method)

        if annotation is not None:
            annotation_key, annotation_vals = next(iter(annotation.items()))
            if annotation_key not in self.adata.obs:
                raise KeyError(f"Unable to access data in [{annotation_key!r}.")
            if not isinstance(annotation_vals, Iterable):
                raise TypeError("`annotation` expected to be dictionary of length 1 with value being a list.")

            adata = self.adata[self.adata.obs[annotation_key].isin(annotation_vals)]
        else:
            adata = self.adata

        adata = adata[~adata.obs[obs_key].isnull()]
        if adata.n_obs == 0:
            raise ValueError(f"`adata.obs[{obs_key!r}]` only contains NaN values.")
        distribution = adata.obs[[obs_key]]

        if isinstance(features, str):
            tfs = TranscriptionFactors.transcription_factors(organism=features)
            features = list(set(tfs).intersection(adata.var_names))
            if len(features) == 0:
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


class NeuralAnalysisMixin(AnalysisMixin[K, B]):
    """Analysis Mixin for Neural OT problems."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def cell_transition(
        self,
        source: Union[K, Tuple[K, AnnData, Union[str, Dict[str, str]]]],
        target: Union[K, Tuple[K, AnnData, Union[str, Dict[str, str]]]],
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "cell",
        batch_size: int = 1024,
        normalize: bool = True,
        k: int = 30,
        key_added: Optional[str] = PlottingDefaults.CELL_TRANSITION,
    ) -> Optional[pd.DataFrame]:
        """
        Compute a grouped transition matrix based on a pseudo-transport matrix.

        This function requires a projection of the velocity field onto existing cells, see
        :meth:`moscot.backends.ott.NeuralOutput.project_transport_matrix`.
        Afterwards, this function computes a transition matrix with entries corresponding to categories, e.g. cell
        types. The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        source
            If of type `K`, key identifying the source distribution.
            If of type :class:`tuple``, the first argument is the key of the source distribution the model was
            trained on, the second argument of :class:`anndata.AnnData`, and the third element one of

                - `str`, then it must refer to a key in :attr:`anndata.AnnData.obsm`.
                - `dict`, then the dictionary stores `attr` (attribute of :class:`anndata.AnnData`) and `key`
                  (key of :class:`anndata.AnnData` ``['{attr}']``).

        target
            If of type `K`, key identifying the target distribution.
            If of type :class:`tuple``, the first argument is the key of the target distribution the model was
            trained on, the second argument of :class:`anndata.AnnData`, and the third element one of

                - `str`, then it must refer to a key in :attr:`anndata.AnnData.obsm`.
                - `dict`, then the dictionary stores `attr` (attribute of :class:`anndata.AnnData`) and `key`
                  (key of :class:`anndata.AnnData` ``['{attr}']``).

        source_groups
            Can be one of the following:

                - if `source_groups` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{source_groups}']``.

                - if `target_groups` is of type :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{source_groups.keys()[0]}']``. The order of the list determines the
                  order in the transition matrix.

        target_groups
            Can be one of the following:

                - if `target_groups` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{target_groups}']``.

                - if `target_groups` is of :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{target_groups.keys()[0]}']``. The order of the list determines the
                  order in the transition matrix.

        forward
            Computes transition from `source_annotations` to `target_annotations` if `True`, otherwise backward.

        aggregation_mode

            - `group`: transition probabilities from the groups defined by `source_annotation` are returned.
            - `cell`: the transition probabilities for each cell are returned.

        batch_size
            Number of data points the matrix-vector products are applied to at the same time.
            The larger, the more memory is required.
        normalize
            If `True` the transition matrix is normalized such that it is stochastic. If `forward` is `True`,
            the transition matrix is row-stochastic, otherwise column-stochastic.
        k
            Number of neighbors used to compute the pseudo-transport matrix if it hasn't been computed by
            :meth:`moscot.backends.ott.output.NeuralSolverOutput`
        key_added
            Key in :attr:`anndata.AnnData.uns` and/or :attr:`anndata.AnnData.obs` where the results
            for the corresponding plotting functions are stored.
            See TODO Notebook for how :mod:`moscot.plotting` works.

        Returns
        -------
        Aggregated transition matrix of cells or groups of cells.

        Notes
        -----
        To visualise the results, see :func:`moscot.pl.cell_transition`.
        """

        if isinstance(source, tuple):
            if len(source) != 2:
                raise ValueError("If `source` is a tuple it must be of length 2.")
            if not isinstance(source[1], AnnData):
                raise TypeError("The first element of the tuple must be of type AnnData.")
            if isinstance(source[2], str):
                source_data = source[1].obsm[source[2]]
            elif isinstance(source[2], dict):
                attr, val = next(iter(source[2]))
                source_data = getattr(source[1], attr)[val]
            else:
                raise TypeError("The second element of the tuple must be of type `str` or `dict`.")
            key_source, adata_src = source[0], source[1]
        else:
            key_source, source_data, adata_src = source, None, self.adata  # type:ignore[attr-defined]

        if isinstance(target, tuple):
            if len(target) != 2:
                raise ValueError("If `source` is a tuple it must be of length 2.")
            if not isinstance(target[1], AnnData):
                raise TypeError("The first element of the tuple must be of type AnnData.")
            if isinstance(target[2], str):
                target_data = target[1].obsm[target[2]]
            elif isinstance(target[2], dict):
                attr, val = next(iter(target[2]))
                target_data = getattr(target[1], attr)[val]
            else:
                raise TypeError("The second element of the tuple must be of type `str` or `dict`.")
            adata_tgt = target[0], target[1]
        else:
            key_target, target_data, adata_tgt = target, None, self.adata  # type:ignore[attr-defined]

        problem = self.problems[key_source, key_target]  # type:ignore[attr-defined]
        try:
            if forward:
                tm_result = problem.solution.transport_matrix
            else:
                tm_result = problem.solution.inverse_transport_matrix
        except ValueError:
            logger.info(f"Projecting transport matrix based on {k} nearest neighbors.")
            tm_result = problem.project_transport_matrix(
                source_data, target_data, forward=forward, save_transport_matrix=True, batch_size=batch_size, k=k
            )

        annotation_key_source, annotations_present_source, annotations_ordered_source = _validate_args_cell_transition(
            adata_src, source_groups
        )
        annotation_key_target, annotations_present_target, annotations_ordered_target = _validate_args_cell_transition(
            adata_src, target_groups
        )
        df_source = (
            adata_src[adata_src.obs[self.temporal_key] == source]  # type:ignore[attr-defined]
            .obs[[annotation_key_source]]
            .copy()
        )
        df_target = (
            adata_tgt[adata_tgt.obs[self.temporal_key] == target]  # type:ignore[attr-defined]
            .obs[[annotation_key_target]]
            .copy()
        )
        annotations_verified_source, annotations_verified_target = _validate_annotations(
            df_source=df_source,
            df_target=df_target,
            source_annotation_key=annotation_key_source,
            target_annotation_key=annotation_key_target,
            source_annotations=annotations_present_source,
            target_annotations=annotations_present_target,
            aggregation_mode="annotation",
            forward=forward,
        )
        tm = pd.DataFrame(
            np.zeros((len(annotations_verified_source), len(annotations_verified_target))),
            index=annotations_verified_source,
            columns=annotations_verified_target,
        )
        for annotation_src in annotations_verified_source:
            for annotation_tgt in annotations_verified_target:
                tm.loc[annotation_src, annotation_tgt] = tm_result[
                    np.ix_((df_source == annotation_src).squeeze(), (df_target == annotation_tgt).squeeze())
                ].sum()

        if normalize:
            tm = tm.div(tm.sum(axis=int(forward)), axis=int(not forward))
        if key_added is not None:
            if aggregation_mode == AggregationMode.CELL and AggregationMode.CELL in self.adata.obs:
                raise KeyError(f"Aggregation is already present in `adata.obs[{aggregation_mode!r}]`.")
            plot_vars = {
                "transition_matrix": tm,
                "source": source,
                "target": target,
                "source_groups": source_groups,
                "target_groups": target_groups,
            }
            Key.uns.set_plotting_vars(
                adata=self.adata,  # type:ignore[attr-defined]
                pl_func_key=PlottingKeys.CELL_TRANSITION,
                key=key_added,
                value=plot_vars,
            )
        return _order_transition_matrix(
            tm=tm,
            source_annotations_verified=annotations_verified_source,
            target_annotations_verified=annotations_verified_target,
            source_annotations_ordered=annotations_ordered_source,
            target_annotations_ordered=annotations_ordered_target,
            forward=forward,
        )

    def push(
        self,
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = PlottingDefaults.PUSH,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Push cells.

        Parameters
        ----------
        %(source)s
        %(target)s
        %(data)s
        %(subset)s
        %(scale_by_marginals)s
        %(key_added_plotting)s
        %(return_all)s
        %(return_data)s
        %(new_adata)s
        %(new_adata_joint_attr)s

        Return
        ------
        %(return_push_pull)s

        """
        result = self._apply(  # type:ignore[attr-defined]
            start=source,
            end=target,
            data=data,
            subset=subset,
            forward=True,
            return_all=return_all or key_added is not None,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )

        if TYPE_CHECKING:
            assert isinstance(result, dict)

        if key_added is not None:
            plot_vars = {
                "temporal_key": self.temporal_key,  # type:ignore[attr-defined]
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.temporal_key)  # type:ignore[attr-defined, misc]
            Key.uns.set_plotting_vars(self.adata, PlottingKeys.PUSH, key_added, plot_vars)  # type:ignore[attr-defined]
        if return_data:
            return result

    def pull(
        self,
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = PlottingDefaults.PULL,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Pull cells.

        Parameters
        ----------
        %(source)s
        %(target)s
        %(data)s
        %(subset)s
        %(scale_by_marginals)s
        %(key_added_plotting)s
        %(return_all)s
        %(return_data)s
        %(new_adata)s
        %(new_adata_joint_attr)s

        Return
        ------
        %(return_push_pull)s

        """
        result = self._apply(  # type:ignore[attr-defined]
            start=source,
            end=target,
            data=data,
            subset=subset,
            forward=False,
            return_all=return_all or key_added is not None,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )
        if TYPE_CHECKING:
            assert isinstance(result, dict)

        if key_added is not None:
            plot_vars = {
                "temporal_key": self.temporal_key,  # type:ignore[attr-defined]
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.temporal_key)  # type:ignore[misc, attr-defined]
            Key.uns.set_plotting_vars(
                self.adata, PlottingKeys.PULL, key_added, plot_vars  # type:ignore[attr-defined]
            )
        if return_data:
            return result

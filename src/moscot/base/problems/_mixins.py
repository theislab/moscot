from __future__ import annotations

import types
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator

import scanpy as sc

from moscot import _constants
from moscot._types import ArrayLike, Numeric_t, Str_Dict_t
from moscot.base.problems._utils import (
    _check_argument_compatibility_cell_transition,
    _correlation_test,
    _get_df_cell_transition,
    _order_transition_matrix,
    _validate_annotations,
    _validate_args_cell_transition,
)
from moscot.base.problems.compound_problem import B, K
from moscot.base.problems.problem import (
    AbstractPushPullAdata,
    AbstractSolutionsProblems,
)
from moscot.plotting._utils import set_plotting_vars
from moscot.utils.data import transcription_factors
from moscot.utils.subset_policy import SubsetPolicy

__all__ = ["AnalysisMixin"]


class AnalysisMixin(Generic[K, B], AbstractPushPullAdata, AbstractSolutionsProblems):
    """Base Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _cell_transition(
        self,
        source: K,
        target: Optional[K],
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

    def _annotation_aggregation_transition(
        self,
        annotations_1: list[Any],
        annotations_2: list[Any],
        df: pd.DataFrame,
        func: Callable[..., ArrayLike],
    ) -> pd.DataFrame:
        n1 = len(annotations_1)
        n2 = len(annotations_2)
        tm_arr = np.zeros((n1, n2))

        # Factorize annotations in df_res_annotation
        codes, uniques = pd.factorize(df.values)
        # Map annotations in 'annotations_2' to indices in 'uniques'
        annotations_in_df_to_idx = {annotation: idx for idx, annotation in enumerate(uniques)}
        annotations_2_codes = [annotations_in_df_to_idx.get(annotation, -1) for annotation in annotations_2]

        for i, subset in enumerate(annotations_1):
            result = func(
                subset=subset,
            )
            # Compute sums over 'codes' weighted by 'result'
            sums = np.bincount(codes, weights=result.squeeze(), minlength=len(uniques))
            dist = [sums[code] if code != -1 else 0 for code in annotations_2_codes]
            tm_arr[i, :] = dist

        return pd.DataFrame(
            tm_arr,
            index=annotations_1,
            columns=annotations_2,
        )

    def _cell_transition_online(
        self,
        key: Optional[str],
        source: K,
        target: Optional[K],
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
        new_annotation_key = "new_annotation"
        df_source = _get_df_cell_transition(
            self.adata,
            [source_annotation_key],
            key,
            source,
        ).rename(columns={source_annotation_key: new_annotation_key})
        df_target = _get_df_cell_transition(
            self.adata if other_adata is None else other_adata,
            [target_annotation_key],
            key if other_adata is None else other_key,
            target,
        ).rename(columns={target_annotation_key: new_annotation_key})
        source_annotations_verified, target_annotations_verified = _validate_annotations(
            df_source=df_source,
            df_target=df_target,
            source_annotation_key=new_annotation_key,
            target_annotation_key=new_annotation_key,
            source_annotations=source_annotations,
            target_annotations=target_annotations,
            aggregation_mode=aggregation_mode,
            forward=forward,
        )
        df_to, df_from = (df_target, df_source) if forward else (df_source, df_target)
        df_to = df_to[new_annotation_key]
        move_op = self.push if forward else self.pull
        move_op_const_kwargs = {
            "source": source,
            "target": target,
            "normalize": True,
            "return_all": False,
            "scale_by_marginals": False,
            "key_added": None,
        }

        if aggregation_mode == "annotation":
            func = partial(
                move_op,
                data=source_annotation_key if forward else target_annotation_key,
                split_mass=False,
                **move_op_const_kwargs,
            )
            tm = self._annotation_aggregation_transition(
                annotations_1=source_annotations_verified if forward else target_annotations_verified,
                annotations_2=target_annotations_verified if forward else source_annotations_verified,
                df=df_to,
                func=func,
            )

        elif aggregation_mode == "cell":
            func = partial(
                move_op,
                data=None,
                split_mass=True,
                **move_op_const_kwargs,
            )
            tm = self._cell_aggregation_transition(
                df_from=df_from,
                df_to=df_to,
                annotations=target_annotations_verified if forward else source_annotations_verified,
                batch_size=batch_size,
                func=func,
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

    def _annotation_mapping(
        self,
        mapping_mode: Literal["sum", "max"],
        annotation_label: str,
        source: K,
        target: K,
        key: str | None = None,
        forward: bool = True,
        other_adata: str | None = None,
        scale_by_marginals: bool = True,
        batch_size: int | None = None,
        cell_transition_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
    ) -> pd.DataFrame:
        if mapping_mode == "sum":
            cell_transition_kwargs = dict(cell_transition_kwargs)
            cell_transition_kwargs.setdefault("aggregation_mode", "cell")  # aggregation mode should be set to cell
            cell_transition_kwargs.setdefault("key", key)
            cell_transition_kwargs.setdefault("source", source)
            cell_transition_kwargs.setdefault("target", target)
            cell_transition_kwargs.setdefault("other_adata", other_adata)
            cell_transition_kwargs.setdefault("forward", not forward)
            cell_transition_kwargs.setdefault("batch_size", batch_size)
            if forward:
                cell_transition_kwargs.setdefault("source_groups", annotation_label)
                cell_transition_kwargs.setdefault("target_groups", None)
                axis = 0  # rows
            else:
                cell_transition_kwargs.setdefault("source_groups", None)
                cell_transition_kwargs.setdefault("target_groups", annotation_label)
                axis = 1  # columns
            out: pd.DataFrame = self._cell_transition(**cell_transition_kwargs)
            return out.idxmax(axis=axis).to_frame(name=annotation_label)
        if mapping_mode == "max":
            out = []
            if forward:
                source_df = _get_df_cell_transition(
                    self.adata,
                    annotation_keys=[annotation_label],
                    filter_key=key,
                    filter_value=source,
                )
                out_len = self.solutions[(source, target)].shape[1]
                batch_size = batch_size if batch_size is not None else out_len
                for batch in range(0, out_len, batch_size):
                    tm_batch: ArrayLike = self.pull(
                        source=source,
                        target=target,
                        data=None,
                        subset=(batch, batch_size),
                        normalize=True,
                        return_all=False,
                        scale_by_marginals=scale_by_marginals,
                        split_mass=True,
                        key_added=None,
                    )
                    v = np.array(tm_batch.argmax(0))
                    out.extend(source_df[annotation_label][v[i]] for i in range(len(v)))

            else:
                target_df = _get_df_cell_transition(
                    self.adata if other_adata is None else other_adata,
                    annotation_keys=[annotation_label],
                    filter_key=key,
                    filter_value=target,
                )
                out_len = self.solutions[(source, target)].shape[0]
                batch_size = batch_size if batch_size is not None else out_len
                for batch in range(0, out_len, batch_size):
                    tm_batch: ArrayLike = self.push(  # type: ignore[no-redef]
                        source=source,
                        target=target,
                        data=None,
                        subset=(batch, batch_size),
                        normalize=True,
                        return_all=False,
                        scale_by_marginals=scale_by_marginals,
                        split_mass=True,
                        key_added=None,
                    )
                    v = np.array(tm_batch.argmax(0))
                    out.extend(target_df[annotation_label][v[i]] for i in range(len(v)))
            categories = pd.Categorical(out)
            return pd.DataFrame(categories, columns=[annotation_label])
        raise NotImplementedError(f"Mapping mode `{mapping_mode!r}` is not yet implemented.")

    def _sample_from_tmap(
        self,
        source: K,
        target: K,
        n_samples: int,
        source_dim: int,
        target_dim: int,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[Numeric_t] = None,
        seed: Optional[int] = None,
    ) -> tuple[list[Any], list[ArrayLike]]:
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
        all_cols_sampled: list[str] = []
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
        self,
        # TODO(@giovp): rename this to 'explicit_steps', pass to policy.plan() and reintroduce (source_key, target_key)
        path: Sequence[tuple[K, K]],
        scale_by_marginals: bool = True,
        **_: Any,
    ) -> LinearOperator:
        """Interpolate transport matrix."""
        if TYPE_CHECKING:
            assert isinstance(self._policy, SubsetPolicy)
        # TODO(@MUCDK, @giovp, discuss what exactly this function should do, seems like it could be more generic)
        fst, *rest = path
        return self.solutions[fst].chain([self.solutions[r] for r in rest], scale_by_marginals=scale_by_marginals)

    def _flatten(self, data: dict[K, ArrayLike], *, key: Optional[str]) -> ArrayLike:
        tmp = np.full(len(self.adata), np.nan)
        for k, v in data.items():
            mask = self.adata.obs[key] == k
            tmp[mask] = np.squeeze(v)
        return tmp

    def _cell_aggregation_transition(
        self,
        df_from: pd.DataFrame,
        df_to: pd.DataFrame,
        annotations: list[Any],
        batch_size: Optional[int],
        func: Callable[..., ArrayLike],
    ) -> pd.DataFrame:

        # Factorize annotations in df_to
        annotations_in_df_to = df_to.values
        codes_to, uniques_to = pd.factorize(annotations_in_df_to)
        # Map annotations in 'annotations' to codes
        annotations_to_code = {annotation: idx for idx, annotation in enumerate(uniques_to)}
        annotations_codes = [annotations_to_code.get(annotation, -1) for annotation in annotations]
        n_annotations = len(annotations)
        n_from_cells = len(df_from)

        if batch_size is None:
            batch_size = n_from_cells

        tm_arr = np.zeros((n_from_cells, n_annotations))
        index = df_from.index

        # Process in batches
        for batch_start in range(0, n_from_cells, batch_size):
            batch_end = min(batch_start + batch_size, n_from_cells)
            subset = (batch_start, batch_end - batch_start)
            result = func(subset=subset)
            # Result shape: (n_to_cells, batch_size)
            # For each cell in the batch, we compute the sum over annotations
            for i in range(batch_end - batch_start):
                cell_distribution = result[:, i]
                # Aggregate over annotations using bincount
                sums = np.bincount(
                    codes_to,
                    weights=cell_distribution,
                    minlength=len(uniques_to),
                )
                # Map sums to annotations_verified_codes
                dist = [sums[code] if code != -1 else 0 for code in annotations_codes]
                tm_arr[batch_start + i, :] = dist

        return pd.DataFrame(tm_arr, index=index, columns=annotations)

    # adapted from:
    # https://github.com/theislab/cellrank/blob/master/cellrank/_utils/_utils.py#L392
    def compute_feature_correlation(
        self,
        obs_key: str,
        corr_method: Literal["pearson", "spearman"] = "pearson",
        significance_method: Literal["fisher", "perm_test"] = "fisher",
        annotation: Optional[dict[str, Iterable[str]]] = None,
        layer: Optional[str] = None,
        features: Optional[Union[list[str], Literal["human", "mouse", "drosophila"]]] = None,
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

    def compute_entropy(
        self,
        source: K,
        target: K,
        forward: bool = True,
        key_added: Optional[str] = "conditional_entropy",
        batch_size: Optional[int] = None,
        c: float = 1e-10,
        **kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """Compute the conditional entropy per cell.

        The conditional entropy reflects the uncertainty of the mapping of a single cell.

        Parameters
        ----------
        source
            Source key.
        target
            Target key.
        forward
            If `True`, computes the conditional entropy of a cell in the source distribution, else the
            conditional entropy of a cell in the target distribution.
        key_added
            Key in :attr:`~anndata.AnnData.obs` where the entropy is stored.
        batch_size
            Batch size for the computation of the entropy. If :obj:`None`, the entire dataset is used.
        c
            Constant added to each row of the transport matrix to avoid numerical instability.
        kwargs
            Kwargs for :func:`~scipy.stats.entropy`.

        Returns
        -------
        :obj:`None` if ``key_added`` is not None. Otherwise, returns a data frame of shape ``(n_cells, 1)`` containing
        the conditional entropy per cell.
        """
        from scipy import stats

        filter_value = source if forward else target
        df = pd.DataFrame(
            index=self.adata[self.adata.obs[self._policy.key] == filter_value, :].obs_names,
            columns=[key_added] if key_added is not None else ["entropy"],
        )
        batch_size = batch_size if batch_size is not None else len(df)
        func = self.push if forward else self.pull
        for batch in range(0, len(df), batch_size):
            cond_dists = func(
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
            df.iloc[range(batch, min(batch + batch_size, len(df))), 0] = stats.entropy(cond_dists + c, **kwargs)
        if key_added is not None:
            self.adata.obs[key_added] = df
        return df if key_added is None else None

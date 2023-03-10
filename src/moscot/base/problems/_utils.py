import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from moscot.base.problems.compound_problem import BaseCompoundProblem
    from moscot.base.problems.problem import BaseProblem

import wrapt
from anndata import AnnData
from statsmodels.stats.multitest import multipletests

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, isspmatrix_csr, spmatrix
from scipy.stats import norm, rankdata

from moscot._constants._constants import AggregationMode, CorrMethod, CorrTestMethod
from moscot._docs._docs import d
from moscot._types import ArrayLike, Str_Dict_t
from moscot.logging import logger


def _validate_annotations_helper(
    df: pd.DataFrame,
    annotation_key: Optional[str] = None,
    annotations: Optional[List[Any]] = None,
    aggregation_mode: Literal["annotation", "cell"] = "annotation",
) -> List[Any]:
    if aggregation_mode == AggregationMode.ANNOTATION:
        if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition(
            assert annotations is not None
        annotations_verified = set(df[annotation_key].cat.categories).intersection(set(annotations))
        if not len(annotations_verified):
            raise ValueError(f"None of `{annotations}` found in the distribution corresponding to `{annotation_key}`.")
        return list(annotations_verified)
    return [None]


def _check_argument_compatibility_cell_transition(
    source_annotation: Str_Dict_t,
    target_annotation: Str_Dict_t,
    key: Optional[str] = None,
    # TODO(MUCDK): unused variable
    other_key: Optional[str] = None,
    other_adata: Optional[AnnData] = None,
    aggregation_mode: Literal["annotation", "cell"] = "annotation",
    forward: bool = False,
    **_: Any,
) -> None:
    if key is None and other_adata is None:
        raise ValueError("Unable to infer distributions, missing `adata` and `key`.")
    if forward and target_annotation is None:
        raise ValueError("No target annotation provided.")
    if not forward and source_annotation is None:
        raise ValueError("No source annotation provided.")
    if (AggregationMode(aggregation_mode) == AggregationMode.ANNOTATION) and (
        source_annotation is None or target_annotation is None
    ):
        raise ValueError(
            f"If aggregation mode is `{AggregationMode.ANNOTATION!r}`, "
            f"source or target annotation in `adata.obs` must be provided."
        )


def _get_df_cell_transition(
    adata: AnnData,
    key: Optional[str] = None,
    key_value: Optional[Any] = None,
    annotation_key: Optional[str] = None,
) -> pd.DataFrame:
    if key is None:
        return adata.obs[[annotation_key]].copy()
    return adata[adata.obs[key] == key_value].obs[[annotation_key]].copy()


def _validate_args_cell_transition(
    adata: AnnData,
    arg: Str_Dict_t,
) -> Tuple[str, List[Any], Optional[List[str]]]:
    if isinstance(arg, str):
        try:
            return arg, adata.obs[arg].cat.categories, None
        except KeyError:
            raise KeyError(f"Unable to fetch data from `adata.obs[{arg!r}]`.") from None
        except AttributeError:
            raise AttributeError(f"Data in `adata.obs[{arg!r}]` is not categorical.") from None

    if isinstance(arg, dict):
        if len(arg) != 1:
            raise ValueError(f"Expected dictionary of length `1`, found `{len(arg)}`.")
        key, val = next(iter(arg.items()))
        if not set(val).issubset(adata.obs[key].cat.categories):
            raise ValueError(f"Not all values `{val}` are present in `adata.obs[{key!r}]`.")
        return key, val, val

    raise TypeError(f"Expected argument to be either `str` or `dict`, found `{type(arg)}`.")


def _get_cell_indices(
    adata: AnnData,
    key: Optional[str] = None,
    key_value: Optional[Any] = None,
) -> pd.Index:
    if key is None:
        return adata.obs.index
    return adata[adata.obs[key] == key_value].obs.index


def _get_categories_from_adata(
    adata: AnnData,
    key: Optional[str] = None,
    key_value: Optional[Any] = None,
    annotation_key: Optional[str] = None,
) -> pd.Series:
    if key is None:
        return adata.obs[annotation_key]
    return adata[adata.obs[key] == key_value].obs[annotation_key]


def _order_transition_matrix_helper(
    tm: pd.DataFrame,
    rows_verified: List[str],
    cols_verified: List[str],
    row_order: Optional[List[str]],
    col_order: Optional[List[str]],
) -> pd.DataFrame:
    if col_order is not None:
        tm = tm[[col for col in col_order if col in cols_verified]]
    tm = tm.T
    if row_order is not None:
        return tm[[row for row in row_order if row in rows_verified]]
    return tm


def _order_transition_matrix(
    tm: pd.DataFrame,
    source_annotations_verified: List[str],
    target_annotations_verified: List[str],
    source_annotations_ordered: Optional[List[str]],
    target_annotations_ordered: Optional[List[str]],
    forward: bool,
) -> pd.DataFrame:
    # TODO(michalk8): simplify
    if target_annotations_ordered is not None or source_annotations_ordered is not None:
        if forward:
            return _order_transition_matrix_helper(
                tm=tm,
                rows_verified=source_annotations_verified,
                cols_verified=target_annotations_verified,
                row_order=source_annotations_ordered,
                col_order=target_annotations_ordered,
            ).T
        return _order_transition_matrix_helper(
            tm=tm,
            rows_verified=target_annotations_verified,
            cols_verified=source_annotations_verified,
            row_order=target_annotations_ordered,
            col_order=source_annotations_ordered,
        )

    if target_annotations_verified == source_annotations_verified:
        annotations_ordered = tm.columns.sort_values()
        if forward:
            return _order_transition_matrix_helper(
                tm=tm,
                rows_verified=source_annotations_verified,
                cols_verified=target_annotations_verified,
                row_order=annotations_ordered,
                col_order=annotations_ordered,
            ).T
        return _order_transition_matrix_helper(
            tm=tm,
            rows_verified=target_annotations_verified,
            cols_verified=source_annotations_verified,
            row_order=annotations_ordered,
            col_order=annotations_ordered,
        )

    return tm if forward else tm.T


@d.dedent
def _correlation_test(
    X: Union[ArrayLike, spmatrix],
    Y: pd.DataFrame,
    feature_names: Sequence[str],
    corr_method: CorrMethod = CorrMethod.PEARSON,
    significance_method: CorrTestMethod = CorrTestMethod.FISCHER,
    confidence_level: float = 0.95,
    n_perms: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Perform a statistical test for correlation between X and .

    Return NaN for genes which don't vary across cells.

    Parameters
    ----------
    X
        Array or sparse matrix of shape ``(n_cells, n_features)`` containing the counts.
    Y
        Data frame of shape ``(n_cells, 1)`` containing the pull/push distribution.
    feature_names
        Sequence of shape ``(n_features,)`` containing the feature names.
    corr_method
        Which type of correlation to compute, options are `pearson`, and `spearman`.
    significance_method
        Method for p-value calculation.
    confidence_level
        Confidence level for the confidence interval calculation. Must be in `[0, 1]`.
    n_perms
        Number of permutations if ``method = 'perm_test'``.
    seed
        Random seed if ``method = 'perm_test'``.
    kwargs
        Keyword arguments for :func:`moscot._utils.parallelize`, e.g. `n_jobs`.

    Returns
    -------
    Dataframe of shape ``(n_genes, 5)`` containing the following columns, one for each lineage:
        - ``corr`` - correlation between the count data and push/pull distributions.
        - ``pval`` - calculated p-values for double-sided test.
        - ``qval`` - corrected p-values using Benjamini-Hochberg method at level `0.05`.
        - ``ci_low`` - lower bound of the ``confidence_level`` correlation confidence interval.
        - ``ci_high`` - upper bound of the ``confidence_level`` correlation confidence interval.
    """
    corr, pvals, ci_low, ci_high = _correlation_test_helper(
        X.T,
        Y.values,
        corr_method=corr_method,
        significance_method=significance_method,
        n_perms=n_perms,
        seed=seed,
        confidence_level=confidence_level,
        **kwargs,
    )
    invalid = (corr < -1) | (corr > 1)
    if np.any(invalid):
        logger.warning(
            f"Found `{np.sum(invalid)}` correlation(s) that are not in `[0, 1]`. "
            f"This usually happens when gene expression is constant across all cells. "
            f"Setting to `NaN`"
        )
        corr[invalid] = np.nan
        pvals[invalid] = np.nan
        ci_low[invalid] = np.nan
        ci_high[invalid] = np.nan

    res = pd.DataFrame(corr, index=feature_names, columns=[f"{c}_corr" for c in Y.columns])
    for idx, c in enumerate(Y.columns):
        p = pvals[:, idx]
        valid_mask = ~np.isnan(p)

        res[f"{c}_pval"] = p
        res[f"{c}_qval"] = np.nan
        if np.any(valid_mask):
            res.loc[np.asarray(feature_names)[valid_mask], f"{c}_qval"] = multipletests(
                p[valid_mask], alpha=0.05, method="fdr_bh"
            )[1]
        res[f"{c}_ci_low"] = ci_low[:, idx]
        res[f"{c}_ci_high"] = ci_high[:, idx]

    # fmt: off
    res = res[[f"{c}_{stat}" for c in Y.columns for stat in ("corr", "pval", "qval", "ci_low", "ci_high")]]
    return res.sort_values(by=[f"{c}_corr" for c in Y.columns], ascending=False)
    # fmt: on


def _correlation_test_helper(
    X: ArrayLike,
    Y: ArrayLike,
    corr_method: CorrMethod = CorrMethod.SPEARMAN,
    significance_method: CorrTestMethod = CorrTestMethod.FISCHER,
    n_perms: Optional[int] = None,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
    **kwargs: Any,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the correlation between rows in matrix ``X`` columns of matrix ``Y``.

    Parameters
    ----------
    X
        Array or matrix of `(M, N)` elements.
    Y
        Array of `(N, K)` elements.
    corr_method
        Which type of correlation to compute, options are `pearson`, and `spearman`.
    significance_method
        Method for p-value calculation.
    n_perms
        Number of permutations if ``method='perm_test'``.
    seed
        Random seed if ``method = 'perm_test'``.
    confidence_level
        Confidence level for the confidence interval calculation. Must be in `[0, 1]`.
    kwargs
        Keyword arguments for :func:`moscot._utils.parallelize`, e.g. `n_jobs`.

    Returns
    -------
    Correlations, p-values, corrected p-values, lower and upper bound of 95% confidence interval.
    """
    from moscot._utils import parallelize

    def perm_test_extractor(res: Sequence[Tuple[ArrayLike, ArrayLike]]) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        pvals, corr_bs = zip(*res)
        if TYPE_CHECKING:
            assert isinstance(n_perms, int)
        pvals = np.sum(pvals, axis=0) / float(n_perms)

        corr_bs = np.concatenate(corr_bs, axis=0)
        corr_ci_low, corr_ci_high = np.quantile(corr_bs, q=ql, axis=0), np.quantile(corr_bs, q=qh, axis=0)

        return pvals, corr_ci_low, corr_ci_high  # type:ignore[return-value]

    if not (0 <= confidence_level <= 1):
        raise ValueError(f"Expected `confidence_level` to be in interval `[0, 1]`, found `{confidence_level}`.")

    n = X.shape[1]  # genes x cells
    ql = 1 - confidence_level - (1 - confidence_level) / 2.0
    qh = confidence_level + (1 - confidence_level) / 2.0

    if issparse(X) and not isspmatrix_csr(X):
        X = csr_matrix(X)

    if corr_method == CorrMethod.SPEARMAN:
        X, Y = rankdata(X, method="average", axis=0), rankdata(Y, method="average", axis=0)
    corr = _pearson_mat_mat_corr_sparse(X, Y) if issparse(X) else _pearson_mat_mat_corr_dense(X, Y)

    if significance_method == CorrTestMethod.FISCHER:
        # see: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_Fisher_transformation
        # for spearman see: https://www.sciencedirect.com/topics/mathematics/spearman-correlation
        mean, se = np.arctanh(corr), 1 / np.sqrt(n - 3)
        z_score = (np.arctanh(corr) - np.arctanh(0)) * np.sqrt(n - 3)

        z = norm.ppf(qh)
        corr_ci_low = np.tanh(mean - z * se)
        corr_ci_high = np.tanh(mean + z * se)
        pvals = 2 * norm.cdf(-np.abs(z_score))

    elif significance_method == CorrTestMethod.PERM_TEST:
        if not isinstance(n_perms, int):
            raise TypeError(f"Expected `n_perms` to be an integer, found `{type(n_perms).__name__}`.")
        if n_perms <= 0:
            raise ValueError(f"Expcted `n_perms` to be positive, found `{n_perms}`.")

        pvals, corr_ci_low, corr_ci_high = parallelize(
            _perm_test,  # type:ignore[arg-type]
            np.arange(n_perms),
            as_array=False,
            unit="permutation",
            extractor=perm_test_extractor,
            **kwargs,
        )(corr, X, Y, seed=seed)

    else:
        raise NotImplementedError(significance_method)

    return corr, pvals, corr_ci_low, corr_ci_high


def _pearson_mat_mat_corr_sparse(
    X: csr_matrix,
    Y: ArrayLike,
) -> ArrayLike:
    n = X.shape[1]
    X_bar = np.reshape(np.array(X.mean(axis=1)), (-1, 1))
    X_std = np.reshape(np.sqrt(np.array(X.power(2).mean(axis=1)) - (X_bar**2)), (-1, 1))

    y_bar = np.reshape(np.mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np.std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)


def _pearson_mat_mat_corr_dense(X: ArrayLike, Y: ArrayLike) -> ArrayLike:
    n = X.shape[1]

    X_bar = np.reshape(np.mean(X, axis=1), (-1, 1))
    X_std = np.reshape(np.std(X, axis=1), (-1, 1))

    y_bar = np.reshape(np.mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np.std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)


def _perm_test(
    ixs: ArrayLike,
    corr: ArrayLike,
    X: Union[ArrayLike, spmatrix],
    Y: ArrayLike,
    seed: Optional[int] = None,
    queue=None,
) -> Tuple[ArrayLike, ArrayLike]:
    rs = np.random.RandomState(None if seed is None else seed + ixs[0])
    cell_ixs = np.arange(X.shape[1])
    pvals = np.zeros_like(corr, dtype=np.float64)
    corr_bs = np.zeros((len(ixs), X.shape[0], Y.shape[1]))  # perms x genes x lineages

    mmc = _pearson_mat_mat_corr_sparse if issparse(X) else _pearson_mat_mat_corr_dense

    for i, _ in enumerate(ixs):
        rs.shuffle(cell_ixs)
        corr_i = mmc(X, Y[cell_ixs, :])
        pvals += np.abs(corr_i) >= np.abs(corr)

        bootstrap_ixs = rs.choice(cell_ixs, replace=True, size=len(cell_ixs))
        corr_bs[i, :, :] = mmc(X[:, bootstrap_ixs], Y[bootstrap_ixs, :])

        if queue is not None:
            queue.put(1)

    if queue is not None:
        queue.put(None)

    return pvals, corr_bs


@wrapt.decorator
def require_solution(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check whether problem has been solved."""
    from moscot.base.problems.compound_problem import BaseCompoundProblem
    from moscot.base.problems.problem import OTProblem

    if isinstance(instance, OTProblem) and instance.solution is None:
        raise RuntimeError("Run `.solve()` first.")
    if isinstance(instance, BaseCompoundProblem) and instance.solutions is None:
        raise RuntimeError("Run `.solve()` first.")
    return wrapped(*args, **kwargs)


@wrapt.decorator
def require_prepare(
    wrapped: Callable[[Any], Any],
    instance: "BaseCompoundProblem",  # type: ignore[type-arg]
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Check whether problem has been prepared."""
    if instance.problems is None:
        raise RuntimeError("Run `.prepare()` first.")
    return wrapped(*args, **kwargs)


@wrapt.decorator
def wrap_prepare(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check and update the state when preparing :class:`moscot.problems.base.OTProblem`."""
    from moscot._constants._constants import ProblemStage
    from moscot.base.problems.problem import (
        ProblemKind,  # TODO: move ENUMs to this file
    )

    instance = wrapped(*args, **kwargs)
    if instance.problem_kind == ProblemKind.UNKNOWN:
        raise RuntimeError("Problem kind was not set after running `.prepare()`.")
    instance._stage = ProblemStage.PREPARED
    return instance


@wrapt.decorator
def wrap_solve(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check and update the state when solving :class:`moscot.problems.base.OTProblem`."""
    from moscot._constants._constants import ProblemStage

    if instance.stage not in (ProblemStage.PREPARED, ProblemStage.SOLVED):
        raise RuntimeError(
            f"Expected problem's stage to be either `'prepared'` or `'solved'`, found `{instance.stage!r}`."
        )
    instance = wrapped(*args, **kwargs)
    instance._stage = ProblemStage.SOLVED
    return instance

import functools
import multiprocessing
import threading
import types
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import joblib as jl
import wrapt

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as st
from statsmodels.stats.multitest import multipletests

from anndata import AnnData

from moscot._logging import logger
from moscot._types import ArrayLike, Str_Dict_t

if TYPE_CHECKING:
    from moscot.base.problems.compound_problem import BaseCompoundProblem
    from moscot.base.problems.problem import BaseProblem


Callback = Callable[..., Any]


def _validate_annotations(
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


def _validate_annotations_helper(
    df: pd.DataFrame,
    annotation_key: Optional[str] = None,
    annotations: Optional[List[Any]] = None,
    aggregation_mode: Literal["annotation", "cell"] = "annotation",
) -> List[Any]:
    if aggregation_mode == "annotation":
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
    if (aggregation_mode == "annotation") and (source_annotation is None or target_annotation is None):
        raise ValueError(
            "If aggregation mode is `'annotation'`, " "source or target annotation in `adata.obs` must be provided."
        )


def _get_df_cell_transition(
    adata: AnnData,
    annotation_keys: List[Optional[str]],
    filter_key: Optional[str] = None,
    filter_value: Optional[Any] = None,
) -> pd.DataFrame:
    if filter_key is not None:
        adata = adata[adata.obs[filter_key] == filter_value]
    return adata.obs[list({ak for ak in annotation_keys if ak is not None})].copy()


def _validate_args_cell_transition(
    adata: AnnData,
    arg: Str_Dict_t,
) -> Tuple[Optional[str], List[Any], Optional[List[str]]]:
    if arg is None:
        return None, [], None
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


def _correlation_test(
    X: Union[ArrayLike, sp.spmatrix],
    Y: pd.DataFrame,
    feature_names: Sequence[str],
    corr_method: Literal["pearson", "spearman"] = "pearson",
    significance_method: Literal["fisher", "perm_test"] = "fisher",
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
        Keyword arguments for parallelization, e.g., `n_jobs`.

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
    corr_method: Literal["pearson", "spearman"] = "spearman",
    significance_method: Literal["fisher", "perm_test"] = "fisher",
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

    if sp.issparse(X):
        X = sp.csr_matrix(X)

    if corr_method == "spearman":
        X, Y = st.rankdata(X, method="average", axis=0), st.rankdata(Y, method="average", axis=0)
    corr = _pearson_mat_mat_corr_sparse(X, Y) if sp.issparse(X) else _pearson_mat_mat_corr_dense(X, Y)

    if significance_method == "fisher":
        # see: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_Fisher_transformation
        # for spearman see: https://www.sciencedirect.com/topics/mathematics/spearman-correlation
        mean, se = np.arctanh(corr), 1 / np.sqrt(n - 3)
        z_score = (np.arctanh(corr) - np.arctanh(0)) * np.sqrt(n - 3)

        z = st.norm.ppf(qh)
        corr_ci_low = np.tanh(mean - z * se)
        corr_ci_high = np.tanh(mean + z * se)
        pvals = 2 * st.norm.cdf(-np.abs(z_score))

    elif significance_method == "perm_test":
        if not isinstance(n_perms, int):
            raise TypeError(f"Expected `n_perms` to be an integer, found `{type(n_perms).__name__}`.")
        if n_perms <= 0:
            raise ValueError(f"Expected `n_perms` to be positive, found `{n_perms}`.")

        pvals, corr_ci_low, corr_ci_high = parallelize(
            _perm_test,  # type: ignore[arg-type]
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
    X: sp.csr_matrix,
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
    X: Union[ArrayLike, sp.spmatrix],
    Y: ArrayLike,
    seed: Optional[int] = None,
    queue=None,
) -> Tuple[ArrayLike, ArrayLike]:
    rs = np.random.RandomState(None if seed is None else seed + ixs[0])
    cell_ixs = np.arange(X.shape[1])
    pvals = np.zeros_like(corr, dtype=np.float64)
    corr_bs = np.zeros((len(ixs), X.shape[0], Y.shape[1]))  # perms x genes x lineages

    mmc = _pearson_mat_mat_corr_sparse if sp.issparse(X) else _pearson_mat_mat_corr_dense

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
    instance = wrapped(*args, **kwargs)
    if instance.problem_kind == "unknown":
        raise RuntimeError("Problem kind was not set after running `.prepare()`.")
    instance._stage = "prepared"
    return instance


@wrapt.decorator
def wrap_solve(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check and update the state when solving :class:`moscot.problems.base.OTProblem`."""
    if instance.stage not in ("prepared", "solved"):
        raise RuntimeError(
            f"Expected problem's stage to be either `'prepared'` or `'solved'`, found `{instance.stage!r}`."
        )
    instance = wrapped(*args, **kwargs)
    instance._stage = "solved"
    return instance


def attributedispatch(func: Optional[Callback] = None, attr: Optional[str] = None) -> Callback:
    """Dispatch a function based on the first value."""

    def dispatch(value: Type[Any]) -> Callback:
        for typ in value.mro():
            if typ in registry:
                return registry[typ]
        return func  # type: ignore[return-value]

    def register(value: Type[Any], func: Optional[Callback] = None) -> Callback:
        if func is None:
            return lambda f: register(value, f)
        registry[value] = func
        return func

    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        typ = type(getattr(instance, str(attr)))
        return dispatch(typ)(instance, *args, **kwargs)

    if func is None:
        return functools.partial(attributedispatch, attr=attr)

    registry: Dict[Type[Any], Callback] = {}
    wrapper.register = register  # type: ignore[attr-defined]
    wrapper.dispatch = dispatch  # type: ignore[attr-defined]
    wrapper.registry = types.MappingProxyType(registry)  # type: ignore[attr-defined]
    functools.update_wrapper(wrapper, func)

    return wrapper


def parallelize(
    callback: Callable[[Any], Any],
    collection: Union[sp.spmatrix, Sequence[Any]],
    n_jobs: Optional[int] = None,
    n_split: Optional[int] = None,
    unit: str = "",
    as_array: bool = True,
    use_ixs: bool = False,
    backend: str = "loky",
    extractor: Optional[Callable[[Any], Any]] = None,
    show_progress_bar: bool = True,
) -> Any:
    """
    Parallelize function call over a collection of elements.

    Parameters
    ----------
    callback
        Function to parallelize.
    collection
        Sequence of items which to chunkify or an already.
    n_jobs
        Number of parallel jobs.
    n_split
        Split ``collection`` into ``n_split`` chunks. If `None`, split into ``n_jobs`` chunks.
    unit
        Unit of the progress bar.
    as_array
        Whether to convert the results not :class:`numpy.ndarray`.
    use_ixs
        Whether to pass indices to the callback.
    backend
        Which backend to use for multiprocessing. See :class:`joblib.Parallel` for valid options.
    extractor
        Function to apply to the result after all jobs have finished.
    show_progress_bar
        Whether to show a progress bar.

    Returns
    -------
    The result depending on ``callable``, ``extractor`` and ``as_array``.
    """
    if show_progress_bar:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            try:
                from tqdm.std import tqdm
            except ImportError:
                tqdm = None
    else:
        tqdm = None

    def update(pbar, queue, n_total):
        n_finished = 0
        while n_finished < n_total:
            try:
                res = queue.get()
            except EOFError as e:
                if not n_finished != n_total:
                    raise RuntimeError(f"Finished only `{n_finished}` out of `{n_total}` tasks.`") from e
                break
            assert res in (None, (1, None), 1)  # (None, 1) means only 1 job
            if res == (1, None):
                n_finished += 1
                if pbar is not None:
                    pbar.update()
            elif res is None:
                n_finished += 1
            elif pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()

    def wrapper(*args, **kwargs):
        if pass_queue and show_progress_bar:
            pbar = None if tqdm is None else tqdm(total=col_len, unit=unit, mininterval=0.125)
            queue = multiprocessing.Manager().Queue()
            thread = threading.Thread(target=update, args=(pbar, queue, len(collections)))
            thread.start()
        else:
            pbar, queue, thread = None, None, None

        res = jl.Parallel(n_jobs=n_jobs, backend=backend)(
            jl.delayed(callback)(
                *((i, cs) if use_ixs else (cs,)),
                *args,
                **kwargs,
                queue=queue,
            )
            for i, cs in enumerate(collections)
        )

        res = np.array(res) if as_array else res
        if thread is not None:
            thread.join()

        return res if extractor is None else extractor(res)

    col_len = collection.shape[0] if sp.issparse(collection) else len(collection)  # type: ignore[union-attr]
    n_jobs = _get_n_cores(n_jobs, col_len)
    if n_split is None:
        n_split = n_jobs

    if sp.issparse(collection):
        n_split = max(1, min(n_split, collection.shape[0]))  # type: ignore
        if n_split == collection.shape[0]:  # type: ignore[union-attr]
            collections = [collection[[ix], :] for ix in range(collection.shape[0])]  # type: ignore
        else:
            step = collection.shape[0] // n_split  # type: ignore[union-attr]
            ixs = [np.arange(i * step, min((i + 1) * step, collection.shape[0])) for i in range(n_split)]  # type: ignore  # noqa: E501
            ixs[-1] = np.append(ixs[-1], np.arange(ixs[-1][-1] + 1, collection.shape[0]))  # type: ignore

            collections = [collection[ix, :] for ix in filter(len, ixs)]  # type:ignore[call-overload]
    else:
        collections = list(filter(len, np.array_split(collection, n_split)))

    n_split = len(collections)
    n_jobs = min(n_jobs, n_split)
    pass_queue = not hasattr(callback, "py_func")  # we'd be inside a numba function

    return wrapper


def _get_n_cores(n_cores: Optional[int], n_jobs: Optional[int]) -> int:
    """
    Make number of cores a positive integer.

    Parameters
    ----------
    n_cores
        Number of cores to use.
    n_jobs
        Number of jobs. This is just used to determine if the collection is a singleton.
        If `1`, always returns `1`.

    Returns
    -------
    Positive integer corresponding to how many cores to use.
    """
    if n_cores == 0:
        raise ValueError("Number of cores cannot be `0`.")
    if n_jobs == 1 or n_cores is None:
        return 1
    if n_cores < 0:
        return multiprocessing.cpu_count() + 1 + n_cores

    return n_cores

from typing import Any, Dict, Tuple, Union, Optional, Callable, Mapping

from matplotlib.figure import Figure
import pandas as pd
import wrapt
import numpy as np

import scanpy as sc
import matplotlib
from moscot.problems.base._mixins import AnalysisMixin
from moscot.problems.base._compound_problem import B, K

@wrapt.decorator
def check_plot_categories(
    wrapped: Callable[[Any], Any], instance: "LabelMixin", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check plotting categories."""
    clusters_labelled = args[0]
    if "palette" not in kwargs:
        if clusters_labelled in self._palette_dict.keys():
            palette = self._palette_dict[clusters_labelled]
        else:
            new_palette = {}
            for cat in self.adata.obs[clusters_labelled].cat.categories:
                if cat in self._PALETTE.keys():
                    new_palette[cat] = self._PALETTE[cat]
                else:
                    new_palette[cat] = np.random.uniform()
            self._palette_dict[clusters_labelled] = new_palette
    _ = wrapped(*args[1:], clusters_labelled=clusters_labelled, palette=self._palette_dict[clusters_labelled], **kwargs)


class LabelMixin(AnalysisMixin[K, B]):
    """Analysis Mixin for all problems involving a temporal dimension."""

    def __init__(self, *args: Any, cmap: matplotlib.colors.ListedColormap = matplotlib.cm.viridis, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None
        self._labelled_batch: Optional[str] = None
        self._batch_to_label: Optional[str] = None
        self._palette: matplotlib.colors.ListedColormap = matplotlib.cm.get_cmap(cmap)
        self._palette_dict: Dict = {}

    def set_marginals(
        self,
        clusters_labelled: str,
        adapted_weights: Dict[str, float],
        key_added: str = "annotated_marginals",
        copy: bool = False,
    ) -> Optional[pd.DataFrame]:  # TODO: move to new mixin
        for k in adapted_weights.keys():
            if (
                k
                not in self.adata[self.adata.obs[self._batch_key] == self._labelled_batch]
                .obs[clusters_labelled]
                .cat.categories
            ):
                raise KeyError(f"TODO: {k} not in `adata.obs[{clusters_labelled}]`.")
        df_tmp = self.adata[self.adata.obs[self._batch_key] == self._labelled_batch].obs[[clusters_labelled]].copy()
        df_tmp["annotated_marginals"] = 1.0
        df_tmp["annotated_marginals"] = df_tmp.apply(
            lambda x: adapted_weights[x[clusters_labelled]] if x[clusters_labelled] in adapted_weights.keys() else 1,
            axis=1,
        )
        if copy:
            return df_tmp[["annotated_marginals"]]
        self.adata.obs["annotated_marginals"] = df_tmp["annotated_marginals"]

    def transition_matrix(self, clusters_source: str, clusters_target: str) -> pd.DataFrame:
        """Compute transition matrix."""
        return self._cell_transition(
            self.batch_key, self.labelled_batch, self.batch_to_label, clusters_source, clusters_target, forward=False
        )

    def get_labels(
        self, clusters_labelled: str, clusters_unlabelled: str, top_k: int = 1, return_scores: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get most likely labels, optionally with top_k."""
        tm = self._cell_transition(
            self.batch_key,
            self.labelled_batch,
            self.batch_to_label,
            clusters_labelled,
            clusters_unlabelled,
            forward=False,
        )
        tmp = tm.index[np.argsort(tm.values, axis=0)[-top_k:, :]]
        dd = {}
        for k in range(top_k):
            dd[k] = {tm.columns.values[i]: tmp[k, i] for i in range(len(tm.columns.values))}
        df_top_k_labels = pd.DataFrame.from_dict(dd)
        df_top_k_labels.iloc[:, ::-1]
        df_top_k_labels = df_top_k_labels.rename(columns={i: str(top_k - i) for i in range(top_k)})
        if not return_scores:
            return df_top_k_labels
        d_2 = {}
        for k, d in dd.items():
            ddd = {}
            for col, ind in d.items():
                ddd[col] = tm.loc[ind, col]
            d_2[k] = ddd
        return df_top_k_labels.iloc[:, ::-1], pd.DataFrame.from_dict(d_2).iloc[:, ::-1].rename(
            columns={i: str(top_k - i) for i in range(top_k)}
        )

    def predictions_to_adata(
        self,
        clusters_unlabelled: str,
        predictions: Union[pd.DataFrame, pd.Series],
        labels_key_added: str = "label_prediction",
        scores: Optional[Union[pd.DataFrame, pd.Series]] = None,
        scores_key_added: Optional[str] = "score_prediction",
    ) -> None:
        """Write predictions to adata."""
        for i in range(len(predictions.columns)):
            _preds = self.adata[self.adata.obs[self.batch_key] == self.batch_to_label].obs.apply(
                lambda x: predictions.loc[x[clusters_unlabelled], str(i + 1)], axis=1
            )
            self.adata.obs[f"{labels_key_added}_{i+1}"] = np.nan
            self.adata.obs[f"{labels_key_added}_{i+1}"] = _preds
        if scores is not None:
            if scores.shape != predictions.shape:
                raise ValueError("TODO: same shape required.")
            for i in range(len(predictions.columns)):
                _scores = self.adata[self.adata.obs[self.batch_key] == self.batch_to_label].obs.apply(
                    lambda x: scores.loc[x[clusters_unlabelled], str(i + 1)], axis=1
                )
                self.adata.obs[f"{scores_key_added}_{i+1}"] = 0
                self.adata.obs[f"{scores_key_added}_{i+1}"] = _scores

    #@check_plot_categories
    def plot_predictions(
        self,
        clusters_labelled: str,
        clusters_unlabelled: str,
        top_k: int,
        labels_key_added: str ="label_prediction",
        scores_key_added: str ="score_prediction",
        label_umap_key: Optional[str] = "X_umap",
        **kwargs: Any,
    ) -> Optional[Figure]:
        """Plot the transferred labels."""
        predictions, scores = self.get_labels(
            clusters_labelled=clusters_labelled,
            clusters_unlabelled=clusters_unlabelled,
            top_k=top_k,
            return_scores=True,
        )
        self.predictions_to_adata(
            clusters_unlabelled=clusters_unlabelled,
            predictions=predictions,
            labels_key_added=labels_key_added,
            scores=scores,
            scores_key_added=scores_key_added,
        )
        labels_pred_keys = [f"{labels_key_added}_{i+1}" for i in range(top_k)]
        scores_pred_keys = [f"{scores_key_added}_{i+1}" for i in range(top_k)]
        self.adata.obs[scores_pred_keys] = self.adata.obs[scores_pred_keys].fillna(0)
        for i in range(top_k):
            #sc.pl.scatter(self.adata, color=labels_pred_keys[i], basis=label_umap_key, palette=clusters_unlabelled, **kwargs)
            #sc.pl.scatter(self.adata, color=scores_pred_keys[i], basis=label_umap_key, palette=clusters_unlabelled, **kwargs)
            sc.pl.scatter(self.adata, color=labels_pred_keys[i], basis=label_umap_key, **kwargs)
            sc.pl.scatter(self.adata, color=scores_pred_keys[i], basis=label_umap_key, **kwargs)

    @property
    def batch_key(self) -> Optional[str]:
        """Return batch key."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, value: Optional[str] = None) -> None:
        # if not is_numeric_dtype(self.adata.obs[value]):
        #    raise TypeError(f"TODO: column must be of numeric data type")
        self._batch_key = value

    @property
    def batch_to_label(self) -> Optional[str]:
        """Return batch key."""
        return self._batch_to_label

    @batch_to_label.setter
    def batch_to_label(self, value: Optional[str] = None) -> None:
        # if not is_numeric_dtype(self.adata.obs[value]):
        #    raise TypeError(f"TODO: column must be of numeric data type")
        self._batch_to_label = value

    @property
    def labelled_batch(self) -> Optional[str]:
        """Return batch key."""
        return self._labelled_batch

    @labelled_batch.setter
    def labelled_batch(self, value: Optional[str] = None) -> None:
        # if not is_numeric_dtype(self.adata.obs[value]):
        #    raise TypeError(f"TODO: column must be of numeric data type")
        self._labelled_batch = value

from typing import Any, Optional, Tuple, Union

from matplotlib.figure import Figure
import pandas as pd

import numpy as np

import scanpy as sc

from moscot.problems.base._mixins import AnalysisMixin
from moscot.problems.base._compound_problem import B, K


class LabelMixin(AnalysisMixin[K, B]):
    """Analysis Mixin for all problems involving a temporal dimension."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None
        self._labelled_batch: Optional[str] = None
        self._batch_to_label: Optional[str] = None

    def transition_matrix(self, clusters_source: str, clusters_target: str) -> pd.DataFrame:
        """Compute transition matrix."""
        return self._cell_transition(
            self.batch_key, self.labelled_batch, self.batch_to_label, clusters_source, clusters_target, forward=False
        )

    def get_labels(
        self, clusters_labelled: str, clusters_unlabelled: str, top_k: int, return_values: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get most likely labels, optionally with ."""
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
        if not return_values:
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
        labels_key_added: str,
        scores: Optional[Union[pd.DataFrame, pd.Series]],
        scores_key_added: str,
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

    def plot_predictions(
        self,
        clusters_labelled: str,
        clusters_unlabelled: str,
        top_k: int,
        labels_key_added="label_prediction",
        scores_key_added="score_prediction",
        label_umap_key: Optional[str] = "X_umap",
        **kwargs: Any,
    ) -> Optional[Figure]:
        """Plot the transferred labels."""
        predictions, scores = self.get_labels(
            clusters_labelled=clusters_labelled,
            clusters_unlabelled=clusters_unlabelled,
            top_k=top_k,
            return_values=True,
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

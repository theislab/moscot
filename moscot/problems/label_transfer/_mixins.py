from optparse import Option
from typing import Any, Dict, Tuple, Union, Mapping, Callable, Optional, Sequence
from typing_extensions import Protocol
from matplotlib import cm
from matplotlib.figure import Figure
import wrapt
import pandas as pd
import matplotlib

import numpy as np

from anndata import AnnData
import scanpy as sc
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation as add_color_palette

from moscot._types import ArrayLike, Numeric_t
from moscot.problems.base._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.problems.base._compound_problem import B, K, ApplyOutput_t
from moscot.problems.base import OTProblem


__all__ = ("LabelMixin")


class LabelMixinProtocol(AnalysisMixinProtocol[K, OTProblem], Protocol[K, B]):
    """Protocol class."""

    adata: AnnData
    problems: Dict[Tuple[K, K], B]
    temporal_key: Optional[str]

    def push(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:  # noqa: D102
        ...

    def pull(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:  # noqa: D102
        ...

    


class LabelMixin(AnalysisMixin[K, OTProblem]):
    """Analysis Mixin for all problems involving a temporal dimension."""

    def __init__(
        self, *args: Any, cmap: matplotlib.colors.ListedColormap = matplotlib.cm.viridis, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._palette: matplotlib.colors.ListedColormap = None
        self._key_labelled: Optional[str] = None
        self._key_unlabelled: Optional[str] = None

    def set_marginals(
        self: LabelMixinProtocol[K, OTProblem],
        clusters_labelled: str,
        adapted_weights: Dict[str, float],
        key_added: str = "annotated_marginals",
        copy: bool = False,
    ) -> Optional[pd.DataFrame]:  # TODO: move to new mixin
        for k in adapted_weights.keys():
            if (
                k
                not in self.adata_labelled
                .obs[clusters_labelled]
                .cat.categories
            ):
                raise KeyError(f"TODO: {k} not in `adata.obs[{clusters_labelled}]`.")
        #df_tmp = self.adata[self.adata.obs[self._batch_key] == self._labelled_batch].obs[[clusters_labelled]].copy()
        #df_tmp["annotated_marginals"] = 1.0
        #df_tmp["annotated_marginals"] = df_tmp.apply(
        #    lambda x: adapted_weights[x[clusters_labelled]] if x[clusters_labelled] in adapted_weights.keys() else 1,
        #    axis=1,
        #)
        
        result = self.adata_labelled.apply(
            lambda x: adapted_weights[x[clusters_labelled]] if x[clusters_labelled] in adapted_weights.keys() else 1,
            axis=1,
        )
        if copy:
            #return df_tmp[["annotated_marginals"]]
            self.adata_labelled.obs[key_added] = result
        self.adata_labelled.obs[key_added] = result

    def transition_matrix(self: LabelMixinProtocol[K, B], clusters_labelled: Union[str, Mapping[str, Sequence[Any]]], clusters_unlabelled: Union[str, Mapping[str, Sequence[Any]]], online: bool=False) -> pd.DataFrame:
        """Compute transition matrix."""
        return self._cell_transition(
            self.key_labelled, self.key_unlabelled, self.subset_labelled, self.subset_unlabelled, clusters_labelled, clusters_unlabelled, forward=False, online=online
        )

    def get_labels(
        self: LabelMixinProtocol[K, B], clusters_labelled: str, clusters_unlabelled: str, top_k: int = 1, return_scores: bool = False, online: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get most likely labels, optionally with top_k."""
        tm = self._cell_transition(
            key=self.key_labelled,
            other_key=self.key_unlabelled,
            key_source=self.subset_labelled,
            key_target=self.subset_unlabelled,
            source_cells=clusters_labelled,
            target_cells=clusters_unlabelled,
            forward=False,
            aggregation="group",
            online=online
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

    def predictions_to_adata( #TODO: continue here
        self: LabelMixinProtocol[K, B],
        clusters_unlabelled: str,
        predictions: Union[pd.DataFrame, pd.Series],
        labels_key_added: str = "label_prediction",
        scores: Optional[Union[pd.DataFrame, pd.Series]] = None,
        scores_key_added: Optional[str] = "score_prediction",
    ) -> None:
        """Write predictions to adata."""
        for i in range(len(predictions.columns)):
            _preds = self.adata_unlabelled.obs.apply(
                lambda x: predictions.loc[x[clusters_unlabelled], str(i + 1)], axis=1
            )
            self.adata_unlabelled.obs[f"{labels_key_added}_{i+1}"] = _preds
        if scores is not None:
            if scores.shape != predictions.shape:
                raise ValueError("TODO: same shape required.")
            for i in range(len(predictions.columns)):
                _scores = self.adata_unlabelled.obs.apply(
                    lambda x: scores.loc[x[clusters_unlabelled], str(i + 1)], axis=1
                )
                self.adata_unlabelled.obs[f"{scores_key_added}_{i+1}"] = _scores

    def plot_predictions(
        self: LabelMixinProtocol[K, B],
        clusters_labelled: str,
        clusters_unlabelled: str,
        top_k: int,
        labels_key_added: str = "label_prediction",
        scores_key_added: str = "score_prediction",
        label_umap_key: Optional[str] = "X_umap",
        online: bool = False,
        **kwargs: Any,
    ) -> Optional[Figure]:
        """Plot the transferred labels."""
        predictions, scores = self.get_labels(
            clusters_labelled=clusters_labelled,
            clusters_unlabelled=clusters_unlabelled,
            top_k=top_k,
            return_scores=True,
            online=online
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
        self.adata_unlabelled.obs[scores_pred_keys] = self.adata_unlabelled.obs[scores_pred_keys].fillna(0)
        self.set_palette(self.adata_labelled, clusters_labelled, kwargs.pop("palette"), kwargs.pop("force_update_colors"))
        for i in range(top_k):
            # sc.pl.scatter(self.adata, color=labels_pred_keys[i], basis=label_umap_key, palette=clusters_unlabelled, **kwargs)
            # sc.pl.scatter(self.adata, color=scores_pred_keys[i], basis=label_umap_key, palette=clusters_unlabelled, **kwargs)
            sc.pl.scatter(self.adata, color=labels_pred_keys[i], basis=label_umap_key, **kwargs)
            sc.pl.scatter(self.adata, color=scores_pred_keys[i], basis=label_umap_key, **kwargs)

    @staticmethod
    def set_palette(
        adata: AnnData,
        key: str,
        palette: Union[str, matplotlib.colors.ListedColormap],
        force_update_colors: bool=False
    ) -> None:
        if key not in adata.obs.columns:
            raise KeyError("TODO: invalid key.")
        add_color_palette(adata, key=key, palette=palette, force_update_colors=force_update_colors)

    @property
    def key_labelled(self) -> Optional[str]:
        """Return key of labelled data."""
        return self._key_labelled

    @key_labelled.setter
    def key_labelled(self, value: Optional[str] = None) -> None:
        self._key_labelled = value

    @property
    def key_unlabelled(self) -> Optional[str]:
        """Return key of unlabelled data."""
        return self._key_unlabelled

    @key_unlabelled.setter
    def key_unlabelled(self, value: Optional[str] = None) -> None:
        self._key_unlabelled = value

    @property
    def subset_labelled(self) -> Optional[Sequence[K]]:
        """Return subset of labelled data."""
        return self._subset_labelled

    @subset_labelled.setter
    def subset_labelled(self, value: Optional[Sequence[K]] = None) -> None:
        self._subset_labelled = value
    
    @property
    def subset_unlabelled(self) -> Optional[Sequence[K]]:
        """Return subset of unlabelled data."""
        return self._subset_unlabelled

    @subset_unlabelled.setter
    def subset_unlabelled(self, value: Optional[Sequence[K]] = None) -> None:
        self._subset_unlabelled = value
    
    """
    @property
    def batch_key(self) -> Optional[str]:
        return self._batch_key

    @batch_key.setter
    def batch_key(self, value: Optional[str] = None) -> None:
        self._batch_key = value

    @property
    def batch_to_label(self) -> Optional[str]:
        return self._batch_to_label

    @batch_to_label.setter
    def batch_to_label(self, value: Optional[str] = None) -> None:
        self._batch_to_label = value

    @property
    def labelled_batch(self) -> Optional[str]:
        return self._labelled_batch

    @labelled_batch.setter
    def labelled_batch(self, value: Optional[str] = None) -> None:
        self._labelled_batch = value"""

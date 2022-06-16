from typing import Any, Dict, List, Tuple, Union, Literal, Mapping, Optional, Sequence, TYPE_CHECKING
import itertools

from sklearn.metrics import pairwise_distances
from typing_extensions import Protocol
import ot
import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._docs import d
from moscot._types import ArrayLike, Numeric_t
from moscot.problems.base._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.problems.base._compound_problem import B, K, ApplyOutput_t


class LabelMixin(AnalysisMixin[K, B]):
    """Analysis Mixin for all problems involving a temporal dimension."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None
        self._labelled_batch: Optional[str] = None
        self._batch_to_label: Optional[str] = None

    def transition_matrix(self, clusters_source: str, clusters_target: str) -> pd.DataFrame:
        return self._cell_transition(self.batch_key, self.labelled_batch, self.batch_to_label, clusters_source, clusters_target, forward=False)

    def get_labels(self, clusters_source: str, clusters_target: str, top_k: int, return_values: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        tm = self._cell_transition(self.batch_key, self.labelled_batch, self.batch_to_label, clusters_source, clusters_target, forward=False)
        tmp=tm.index[np.argsort(tm.values, axis=0)[-top_k:, :]]
        dd = {}
        for k in range(top_k):
            dd[k] = {tm.columns.values[i]: tmp[k,i] for i in range(len(tm.columns.values))}
        df_top_k_labels = pd.DataFrame.from_dict(dd)
        if not return_values:
            return df_top_k_labels
        d_2 = {}
        for k, d in dd.items():
            ls = []
            for col, ind in d.items():
                ls.append(tm.loc[ind, col])
            d_2[k] = ls
        return df_top_k_labels.iloc[:, ::-1], pd.DataFrame.from_dict(d_2).iloc[:, ::-1]


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

    def plot_annotation(self, )



from typing import Any, Optional, Union
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from sklearn.preprocessing import normalize
from anndata import AnnData
from numpy.typing import ArrayLike

from moscot.tmp.solvers._data import Tag, TaggedArray
from moscot.tmp._costs import BaseLoss



@dataclass(frozen=True)
class AnnDataPointer:
    adata: AnnData
    attr: str
    key: Optional[str] = None
    use_raw: Optional[bool] = False
    # TODO(michalk8): determine whether this needs to really be here or can be inferred purely from loss/attr
    tag: Tag = Tag.POINT_CLOUD
    # TODO(michalk8): handle custom losses (barcode/tree distances)
    # TODO(michalk8): propagate custom losses in TaggedArray
    loss: Optional[Union[str, ArrayLike, BaseLoss]] = "sqeucl"

    def create(self, **kwargs: Any) -> TaggedArray:
        if not hasattr(self.adata, self.attr):
            raise AttributeError("TODO: invalid attribute")
        container = getattr(self.adata, self.attr)

        if self.key is None:
            # TODO(michalk8): check if array-like
            # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
            return TaggedArray(container, tag=self.tag)
        if self.key not in container:
            raise KeyError(f"TODO: unable to find `adata.{self.attr}['{self.key}']`.")
        container = container[self.key]

        # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
        if self.tag == Tag.COST_MATRIX and isinstance(self.loss, str):
            loss = BaseLoss(kind=self.loss).create(**kwargs)
        elif self.tag == Tag.COST_MATRIX:
            loss = self.loss

        return TaggedArray(container, tag=self.tag, loss=loss)


@dataclass(frozen=True)
class AnnDataMarginal:
    adata: AnnData
    attr: Optional[str] = None
    key: Optional[str] = None

    def create(self, **kwargs: Any) -> npt.ArrayLike:

        def ensure_1D(arr: npt.ArrayLike) -> npt.ArrayLike:
            if arr.ndim != 1:
                raise ValueError("TODO: expected 1D")
            return arr.reshape(-1, 1)

        if self.attr is None:
            return np.ones(self.adata.n_obs)/self.adata.n_obs

        if not hasattr(self.adata, self.attr):
            raise AttributeError("TODO: invalid attribute")
        container = getattr(self.adata, self.attr)

        if self.key is None:
            # TODO(michalk8): check if array-like
            return ensure_1D(np.array(normalize(container, norm="l1")))
        if self.key not in container:
            raise KeyError(f"TODO: unable to find `adata.{self.attr}['{self.key}']`.")
        return normalize(ensure_1D(np.array(container[self.key])), norm="l1")

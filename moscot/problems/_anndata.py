from typing import Any, Optional
from dataclasses import dataclass

from scipy.sparse import issparse
from sklearn.preprocessing import scale
import scipy

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot._utils import _get_backend_losses
from moscot.costs._costs import __all__ as moscot_losses, BaseLoss
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ("AnnDataPointer",)


@dataclass(frozen=True)
class AnnDataPointer:
    """AnnData Pointer."""

    adata: AnnData
    attr: str
    key: Optional[str] = None
    use_raw: Optional[bool] = False
    # TODO(michalk8): determine whether this needs to really be here or can be inferred purely from loss/attr
    tag: Tag = Tag.POINT_CLOUD
    loss: str = "Euclidean"
    # TODO(MUCDK): handle Grid cost. this must be a sequence:
    # https://github.com/google-research/ott/blob/b1adc2894b76b7360f639acb10181f2ce97c656a/ott/geometry/grid.py#L55

    def create(self, **kwargs: Any) -> TaggedArray:  # I rewrote the logic a bit as this way I find it more readable
        """Create AnnData pointer."""

        def ensure_2D(arr: npt.ArrayLike, *, allow_reshape: bool = True) -> np.ndarray:
            arr = np.asarray(arr)
            arr = np.reshape(arr, (-1, 1)) if (allow_reshape and arr.ndim == 1) else arr
            if arr.ndim != 2:
                raise ValueError("TODO: expected 2D")
            return arr

        max_scale = kwargs.pop("max_scale", None)
        standard_scale = kwargs.pop("standard_scale", None)
        if self.tag == Tag.COST_MATRIX:
            if self.loss in moscot_losses:
                container = BaseLoss(kind=self.loss).create(**kwargs)
                return TaggedArray(container, tag=self.tag, loss=None)
            if not hasattr(self.adata, self.attr):
                raise AttributeError("TODO: invalid attribute")
            container = getattr(self.adata, self.attr)

            if issparse(container):
                container = container.A
            if self.key is None:
                return TaggedArray(ensure_2D(container), tag=self.tag, loss=None)
            else:
                if self.key not in container:
                    raise KeyError(f"TODO: unable to find `adata.{self.attr}['{self.key}']`.")
                container = ensure_2D(container[self.key])
                # TODO(michalk8): check if array-like
                # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
                return TaggedArray(container, tag=self.tag, loss=None)
            # TODO(michalk8): not reachable...
            raise ValueError(f"The loss `{self.loss}` is not implemented. Please provide your own cost matrix.")

        backend_losses = _get_backend_losses(**kwargs)  # TODO: put in registry
        if self.loss not in backend_losses.keys():
            raise ValueError(f"The loss `{self.loss}` is not implemented. Please provide your own cost matrix.")
        if not hasattr(self.adata, self.attr):
            raise AttributeError("TODO: invalid attribute")
        container = getattr(self.adata, self.attr)
        if scipy.sparse.issparse(container):
            container = container.A
        if max_scale:
            container /= container.max() * container.shape[1]
        if standard_scale:
            container = scale(container)
        if self.key is None:
            # TODO(michalk8): check if array-like
            # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
            return TaggedArray(container, tag=self.tag, loss=backend_losses[self.loss])
        if self.key not in container:
            raise KeyError(f"TODO: unable to find `adata.{self.attr}['{self.key}']`.")
        container = container[self.key]
        return TaggedArray(container, tag=self.tag, loss=backend_losses[self.loss])

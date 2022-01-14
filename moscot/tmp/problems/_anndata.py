from typing import Any, Optional, Union
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from anndata import AnnData
from numpy.typing import ArrayLike

from moscot.tmp.solvers._data import Tag, TaggedArray
from moscot.tmp._costs import BaseLoss
from moscot.tmp._costs import __all__ as moscot_losses
from moscot.tmp.utils import _get_backend_losses


@dataclass(frozen=True)
class AnnDataPointer:
    adata: AnnData
    attr: str
    key: Optional[str] = None
    use_raw: Optional[bool] = False
    # TODO(michalk8): determine whether this needs to really be here or can be inferred purely from loss/attr
    tag: Tag = Tag.POINT_CLOUD
    loss: str = "Euclidean"
    #TODO(MUCDK): handle Grid cost. this must be a sequence: https://github.com/google-research/ott/blob/b1adc2894b76b7360f639acb10181f2ce97c656a/ott/geometry/grid.py#L55

    def create(self, **kwargs: Any) -> TaggedArray:  # I rewrote the logic a bit as this way I find it more readable
        if self.tag != Tag.COST_MATRIX:
            backend_losses = _get_backend_losses(**kwargs)  # TODO: put in registry
            if self.loss not in backend_losses.keys():
                if self.loss in moscot_losses:
                    container = BaseLoss(kind=self.loss).create(**kwargs)
                    return TaggedArray(container, tag=self.tag, loss=None)
                raise ValueError(f"The loss `{self.loss}` is not implemented. Please provide your own cost matrix.")
            if not hasattr(self.adata, self.attr):
                raise AttributeError("TODO: invalid attribute")
            container = getattr(self.adata, self.attr)
            if self.key is None:
                # TODO(michalk8): check if array-like
                # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
                return TaggedArray(container, tag=self.tag, loss=backend_losses[self.loss])
            if self.key not in container:
                raise KeyError(f"TODO: unable to find `adata.{self.attr}['{self.key}']`.")
            container = container[self.key]
            return TaggedArray(container, tag=self.tag, loss=backend_losses[self.loss])

        if not hasattr(self.adata, self.attr):
            raise AttributeError("TODO: invalid attribute")
        container = getattr(self.adata, self.attr)
        if self.key is None:
            # TODO(michalk8): check if array-like
            # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
            return TaggedArray(container, tag=self.tag, loss=None)
        if self.key not in container:
            raise KeyError(f"TODO: unable to find `adata.{self.attr}['{self.key}']`.")
        container = container[self.key]
        return TaggedArray(container, tag=self.tag, loss=None)

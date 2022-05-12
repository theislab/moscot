from types import MappingProxyType
from typing import Any, Mapping, Optional
from dataclasses import dataclass

from scipy.sparse import issparse
import scipy

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot._docs import d
from moscot._utils import _get_backend_losses
from moscot.costs._costs import BaseLoss
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ("AnnDataPointer",)


@d.dedent
@dataclass(frozen=True)
class AnnDataPointer:
    """
    Class handling the data internally.

    This class handles the data needed to define the cost function in the OT problem.

    Parameters
    ----------
    %(adata)s
    attr
        attribute of :class:`anndata.AnnData` where data is stored
    key
        key of :class:`anndata.AnnData` ``['{key}']`` where the data is stored
    use_raw
        TODO: remove?
    tag
        tag indicating which way the data is stored, valid values are
            - `point_cloud`
            - `cost`
            - `kernel`
            - `grid`
    loss
        loss provided by :class:`moscot.costs` or by `backend`. In the former case the `Tag` must be `cost_matrix`,
        in the latter the `Tag` must be `point_cloud`
    loss_kwargs
        keyword arguments for :meth:`moscot.costs._costs.BaseLoss.create`
    """

    adata: AnnData
    attr: str
    key: Optional[str] = None
    use_raw: Optional[bool] = False
    # TODO(michalk8): determine whether this needs to really be here or can be inferred purely from loss/attr
    tag: Tag = Tag.POINT_CLOUD
    loss: str = "Euclidean"
    loss_kwargs: Mapping[str, Any] = MappingProxyType({})
    # TODO(MUCDK): handle Grid cost. this must be a sequence:
    # https://github.com/google-research/ott/blob/b1adc2894b76b7360f639acb10181f2ce97c656a/ott/geometry/grid.py#L55

    def create(self) -> TaggedArray:  # I rewrote the logic a bit as this way I find it more readable
        """Create."""

        def ensure_2D(arr: npt.ArrayLike, *, allow_reshape: bool = True) -> np.ndarray:
            arr = np.asarray(arr)
            arr = np.reshape(arr, (-1, 1)) if (allow_reshape and arr.ndim == 1) else arr
            if arr.ndim != 2:
                raise ValueError("TODO: expected 2D")
            return arr

        if self.tag == Tag.COST_MATRIX:
            if self.loss is not None:
                cost_matrix = BaseLoss.create(kind=self.loss, adata=self.adata, attr=self.attr, key=self.key)(
                    **self.loss_kwargs
                )
                return TaggedArray(cost_matrix, tag=self.tag, loss=None)
            if not hasattr(self.adata, self.attr):
                raise AttributeError(f"TODO: invalid attribute: {self.attr}")
            container = getattr(self.adata, self.attr)
            if issparse(container):
                container = container.A
            if self.key is None:
                return TaggedArray(ensure_2D(container), tag=self.tag, loss=None)
            if self.key not in container:
                raise KeyError(f"TODO: unable to find `adata.{self.attr}['{self.key}']`.")
            if issparse(container[self.key]):
                container = ensure_2D(container[self.key].A)
            else:
                container = ensure_2D(container[self.key])
            # TODO(michalk8): check if array-like
            # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
            return TaggedArray(container, tag=self.tag, loss=None)

        # TODO(@michalk) handle backend losses
        backend_losses = _get_backend_losses()  # TODO: put in registry, provide kwargs
        if not hasattr(self.adata, self.attr):
            raise AttributeError(f"TODO: invalid attribute: {self.attr}")
        container = getattr(self.adata, self.attr)
        if scipy.sparse.issparse(container):
            return TaggedArray(
                container.A, tag=self.tag, loss=backend_losses[self.loss]
            )  # TODO(@Mmichalk8) propagate loss_kwargs
        if self.key is None:
            return TaggedArray(container, tag=self.tag, loss=backend_losses[self.loss])
        if self.key not in container:
            raise KeyError(f"TODO: unable to find `adata.{self.attr}['{self.key}']`.")
        container = container[self.key]
        return TaggedArray(container.A if scipy.sparse.issparse(container) else container, tag=self.tag, loss=backend_losses[self.loss])

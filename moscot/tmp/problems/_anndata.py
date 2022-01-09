from typing import Any, Optional
from dataclasses import dataclass

from anndata import AnnData

from moscot.tmp.solvers._data import Tag, TaggedArray


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
    loss: Optional[str] = "sqeucl"

    def create(self, **kwargs: Any) -> TaggedArray:
        if not hasattr(self.adata, self.attr):
            raise AttributeError("TODO: invalid attribute")
        container = getattr(self.adata, self.attr)

        if self.key is None:
            # TODO(michalk8): check if array-like
            # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
            return TaggedArray(container, tag=self.tag)
        if self.key not in container:
            raise KeyError(f"TODO: unable to find `adata.{self.attr}[{self.key}]`.")
        container = container[self.key]

        # TODO(michalk8): here we'd construct custom loss (BC/graph distances)
        return TaggedArray(container, tag=self.tag)

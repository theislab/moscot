from typing import Any, Mapping, Optional
from dataclasses import dataclass

from anndata import AnnData

from moscot.tmp.solvers._data import TaggedArray


@dataclass(frozen=True)
class AnnDataPointer:
    adata: AnnData
    attr: str
    key: Optional[str] = None
    use_raw: Optional[bool] = None
    loss: str = "TODO"  # custom losses

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AnnDataPointer":
        pass

    def create(self, **kwargs: Any) -> TaggedArray:
        pass

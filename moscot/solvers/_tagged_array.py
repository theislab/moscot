from typing import Any, Tuple, Union, Literal, Callable, Optional
from dataclasses import dataclass

import scipy.sparse as sp

from anndata import AnnData

from moscot._types import CostFn_t, ArrayLike
from moscot._logging import logger
from moscot._constants._enum import ModeEnum

__all__ = ["Tag", "TaggedArray", "get_cost_function"]

from moscot.costs._costs import BaseLoss


def get_cost_function(cost: str, *, backend: Literal["ott"] = "ott", **kwargs: Any) -> Callable[..., Any]:
    if backend == "ott":
        from moscot.backends.ott._solver import Cost

        return Cost(cost)(**kwargs)

    raise NotImplementedError(f"Backend `{backend}` is not yet implemented.")


class Tag(ModeEnum):
    """Tag of :class:`moscot.solvers._tagged_array.TaggedArray`."""

    COST_MATRIX = "cost"
    KERNEL = "kernel"
    POINT_CLOUD = "point_cloud"
    GRID = "grid"


@dataclass(frozen=True, repr=True)
class TaggedArray:
    """Tagged Array."""

    # passed to solver._prepare_input
    data: ArrayLike
    data_y: Optional[ArrayLike] = None
    tag: Tag = Tag.POINT_CLOUD
    cost: Optional[Union[str, Callable[..., Any]]] = None

    @property
    def is_cost_matrix(self) -> bool:
        """Whether :attr:`data` is a cost matrix."""
        return self.tag == Tag.COST_MATRIX

    @property
    def is_kernel(self) -> bool:
        """Whether :attr:`data` is a kernel matrix."""
        return self.tag == Tag.KERNEL

    @property
    def is_point_cloud(self) -> bool:
        """Whether :attr:`data` is a point cloud."""
        return self.tag == Tag.POINT_CLOUD

    @staticmethod
    def _extract_data(
        adata: AnnData,
        *,
        attr: Literal["X", "obsp", "obsm", "layers", "uns"],
        key: Optional[str] = None,
    ) -> ArrayLike:
        modifier = f"adata.{attr}" if key is None else f"adata.{attr}[{key!r}]"

        try:
            data = getattr(adata, attr)
        except AttributeError:
            raise AttributeError(f"Annotated data object has no attribute `{attr}`.") from None

        try:
            if key is not None:
                data = data[key]
        except KeyError:
            raise KeyError(f"Unable to fetch data from `{modifier}`.")
        except IndexError:
            raise IndexError(f"Unable to fetch data from `{modifier}`.")

        if sp.issparse(data):
            logger.warning(f"Densifying data in `{modifier}`")
            data = data.A
        if data.ndim != 2:
            raise ValueError(f"Expected `{modifier}` to have `2` dimensions, found `{data.ndim}`.")

        return data

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        dist_key: Union[Any, Tuple[Any, Any]],
        attr: Literal["X", "obsp", "obsm", "layers", "uns"],
        tag: Tag = Tag.POINT_CLOUD,
        key: Optional[str] = None,
        cost: CostFn_t = "sq_euclidean",
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> "TaggedArray":
        if tag == Tag.COST_MATRIX:
            if cost == "custom":  # our custom cost functions
                data = cls._extract_data(adata, attr=attr, key=key)
                return cls(data=data, tag=Tag.COST_MATRIX, cost=None)

            cost_matrix = BaseLoss.create(
                kind=cost,  # type: ignore[arg-type]
                adata=adata,
                attr=attr,
                key=key,
                dist_key=dist_key,
            )(**kwargs)
            return cls(data=cost_matrix, tag=Tag.COST_MATRIX, cost=None)

        # tag is either a point cloud or a kernel
        data = cls._extract_data(adata, attr=attr, key=key)
        cost_fn = get_cost_function(cost, backend=backend, **kwargs)
        return cls(data=data, tag=tag, cost=cost_fn)

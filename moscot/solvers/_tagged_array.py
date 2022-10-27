from typing import Any, Union, Literal, Callable, Optional
from dataclasses import dataclass

import scipy.sparse as sp

from anndata import AnnData

from moscot._types import ArrayLike
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
    tag: Tag = Tag.POINT_CLOUD  # TODO(michalk8): in post_init, do check if it's correct type/loss provided
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
        key_specified = key is not None
        modifier = f"adata.{attr}[{key!r}]" if key_specified else f"adata.{attr}"

        try:
            data = getattr(adata, attr)
        except AttributeError:
            raise AttributeError(f"Annotated data object has no attribute `{attr}`.") from None

        try:
            data = data[key]
        except KeyError:
            if key_specified:
                raise KeyError(f"Unable to fetch data from `{modifier}`.")
        except IndexError:
            if key_specified:
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
        tag: Tag,
        attr: Literal["X", "obsp", "obsm", "layers", "uns"],
        key: Optional[str] = None,
        cost: Union[str, Literal["barcode_distance", "leaf_distance", "custom"]] = "SqEuclidean",
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
            )(**kwargs)
            return cls(data=cost_matrix, tag=Tag.COST_MATRIX, cost=None)

        # tag is either a point cloud or a kernel
        data = cls._extract_data(adata, attr=attr, key=key)
        cost_fn = get_cost_function(cost, backend=backend, **kwargs)
        return cls(data=data, tag=tag, cost=cost_fn)

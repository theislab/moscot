import enum
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from anndata import AnnData

from moscot._logging import logger
from moscot._types import ArrayLike, CostFn_t
from moscot.costs import get_cost

__all__ = ["Tag", "TaggedArray"]


@enum.unique
class Tag(str, enum.Enum):
    """Tag in the :class:`~moscot.utils.tagged_array.TaggedArray`."""

    COST_MATRIX = "cost_matrix"  #: Cost matrix.
    KERNEL = "kernel"  #: Kernel matrix.
    POINT_CLOUD = "point_cloud"  #: Point cloud.


@dataclass(frozen=True, repr=True)
class TaggedArray:
    """Interface to interpret array-like data for :mod:`moscot.solvers`.

    It is used to extract array-like data stored in :class:`~anndata.AnnData` and interpret it as either
    :attr:`cost matrix <is_cost_matrix>`, :attr:`kernel matrix <is_kernel>` or
    a :attr:`point cloud <is_point_cloud>`, depending on the :attr:`tag`.

    Parameters
    ----------
    data_src
        Source data.
    data_tgt
        Target data.
    tag
        How to interpret :attr:`data_src` and :attr:`data_tgt`.
    cost
        Cost function when ``tag = 'point_cloud'``.
    """

    data_src: ArrayLike  #: Source data.
    data_tgt: Optional[ArrayLike] = None  #: Target data.
    tag: Tag = Tag.POINT_CLOUD  #: How to interpret :attr:`data_src` and :attr:`data_tgt`.
    cost: Optional[Union[str, Callable[..., Any]]] = None  #: Cost function when ``tag = 'point_cloud'``.

    @staticmethod
    def _extract_data(
        adata: AnnData,
        *,
        attr: Literal["X", "obsp", "obsm", "layers", "uns"],
        key: Optional[str] = None,
    ) -> ArrayLike:
        modifier = f"adata.{attr}" if key is None else f"adata.{attr}[{key!r}]"
        data = getattr(adata, attr)

        try:
            if key is not None:
                data = data[key]
        except KeyError:
            raise KeyError(f"Unable to fetch data from `{modifier}`.") from None
        except IndexError:
            raise IndexError(f"Unable to fetch data from `{modifier}`.") from None

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
        """Create tagged array from :class:`~anndata.AnnData`.

        .. warning::
            Sparse arrays will be always densified.

        Parameters
        ----------
        adata
            Annotated data object.
        dist_key
            Key which determines into which source/target subset ``adata`` belongs.
        attr
            Attribute of :class:`~anndata.AnnData` used when extracting/computing the cost.
        tag
            Tag used to interpret the extracted data.
        key
            Key in the ``attr`` of :class:`~anndata.AnnData` used when extracting/computing the cost.
        cost
            Cost function to apply to the extracted array, depending on ``tag``:

            - if ``tag = 'point_cloud'``, it is extracted from the ``backend``.
            - if ``tag = 'cost'`` or ``tag = 'kernel'``, and ``cost = 'custom'``,
              the extracted array is already assumed to be a cost/kernel matrix.
              Otherwise, :class:`~moscot.base.cost.BaseCost` is used to compute the cost matrix.
        backend
            Which backend to use, see :func:`~moscot.backends.utils.get_available_backends`.
        kwargs
            Keyword arguments for the :class:`~moscot.base.cost.BaseCost` or any backend-specific cost.

        Returns
        -------
        The tagged array.
        """
        if tag == Tag.COST_MATRIX:
            if cost == "custom":  # our custom cost functions
                modifier = f"adata.{attr}" if key is None else f"adata.{attr}[{key!r}]"
                data = cls._extract_data(adata, attr=attr, key=key)
                if np.any(data < 0):
                    raise ValueError(f"Cost matrix in `{modifier}` contains negative values.")
                return cls(data_src=data, tag=Tag.COST_MATRIX, cost=None)

            cost_fn = get_cost(cost, backend="moscot", adata=adata, attr=attr, key=key, dist_key=dist_key)
            cost_matrix = cost_fn(**kwargs)
            return cls(data_src=cost_matrix, tag=Tag.COST_MATRIX, cost=None)

        # tag is either a point cloud or a kernel
        data = cls._extract_data(adata, attr=attr, key=key)
        cost_fn = get_cost(cost, backend=backend, **kwargs)
        return cls(data_src=data, tag=tag, cost=cost_fn)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the cost matrix."""
        if self.tag == Tag.POINT_CLOUD:
            x, y = self.data_src, (self.data_src if self.data_tgt is None else self.data_tgt)
            return x.shape[0], y.shape[0]

        return self.data_src.shape  # type: ignore[return-value]

    @property
    def is_cost_matrix(self) -> bool:
        """Whether :attr:`data_src` is a cost matrix."""
        return self.tag == Tag.COST_MATRIX

    @property
    def is_kernel(self) -> bool:
        """Whether :attr:`data_src` is a kernel matrix."""
        return self.tag == Tag.KERNEL

    @property
    def is_point_cloud(self) -> bool:
        """Whether :attr:`data_src` (and optionally) :attr:`data_tgt` is a point cloud."""
        return self.tag == Tag.POINT_CLOUD

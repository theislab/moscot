import enum
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import scipy.sparse as sp

from anndata import AnnData

from moscot._logging import logger
from moscot._types import ArrayLike, CostFn_t, OttCostFn_t
from moscot.costs import get_cost

K = TypeVar("K", bound=Hashable)

__all__ = ["Tag", "TaggedArray", "DistributionContainer", "DistributionCollection"]


@enum.unique
class Tag(str, enum.Enum):
    """Tag in the :class:`~moscot.utils.tagged_array.TaggedArray`."""

    COST_MATRIX = "cost_matrix"  #: Cost matrix.
    KERNEL = "kernel"  #: Kernel matrix.
    POINT_CLOUD = "point_cloud"  #: Point cloud.
    GRAPH = "graph"  #: Graph distances, means [n+m, n+m] transport matrix.


@dataclass(frozen=False, repr=True)
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
        densify: bool = False,
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

        if sp.issparse(data) and densify:
            logger.warning(f"Densifying data in `{modifier}`")
            data = data.toarray()
        if data.ndim != 2:
            raise ValueError(f"Expected `{modifier}` to have `2` dimensions, found `{data.ndim}`.")

        return data

    def _set_cost(
        self,
        cost: CostFn_t = "sq_euclidean",
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> "TaggedArray":
        if cost == "custom":
            raise ValueError("Custom cost functions are handled in `TaggedArray.from_adata`.")
        if cost != "geodesic":
            cost = get_cost(cost, backend=backend, **kwargs)
        self.cost = cost
        return self

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
            Sparse arrays will be densified except when ``tag = 'graph'``.

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
            - if ``tag = 'graph'`` the ``cost`` has to be ``'geodesic'``.
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
        if tag == Tag.GRAPH:
            if cost == "geodesic":
                dist_key = f"{dist_key[0]}_{dist_key[1]}" if isinstance(dist_key, tuple) else dist_key
                data = cls._extract_data(adata, attr=attr, key=f"{dist_key}_{key}", densify=False)
                return cls(data_src=data, tag=Tag.GRAPH, cost="geodesic")
            raise ValueError(f"Expected `cost=geodesic`, found `{cost}`.")
        if tag == Tag.COST_MATRIX:
            if cost == "custom":  # our custom cost functions
                modifier = f"adata.{attr}" if key is None else f"adata.{attr}[{key!r}]"
                data = cls._extract_data(adata, attr=attr, key=key, densify=True)
                if np.any(data < 0):
                    raise ValueError(f"Cost matrix in `{modifier}` contains negative values.")
                return cls(data_src=data, tag=Tag.COST_MATRIX, cost=None)

            cost_fn = get_cost(cost, backend="moscot", adata=adata, attr=attr, key=key, dist_key=dist_key)
            cost_matrix = cost_fn(**kwargs)
            return cls(data_src=cost_matrix, tag=Tag.COST_MATRIX, cost=None)

        # tag is either a point cloud or a kernel
        data = cls._extract_data(adata, attr=attr, key=key, densify=True)
        cost_fn = get_cost(cost, backend=backend, **kwargs)
        return cls(data_src=data, tag=tag, cost=cost_fn)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the cost matrix."""
        if self.tag == Tag.POINT_CLOUD:
            x, y = self.data_src, (self.data_src if self.data_tgt is None else self.data_tgt)
            return x.shape[0], y.shape[0]

        return self.data_src.shape

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

    @property
    def is_graph(self) -> bool:
        """Whether :attr:`data_src` is a graph."""
        return self.tag == Tag.GRAPH


@dataclass(frozen=True, repr=True)
class DistributionContainer:
    """Data container for OT problems involving more than two distributions.

    TODO

    Parameters
    ----------
    xy
        Distribution living in a shared space.
    xx
        Distribution living in an incomparable space.
    a
        Marginals when used as source distribution.
    b
        Marginals when used as target distribution.
    conditions
        Conditions for the distributions.
    cost_xy
        Cost function when in the shared space.
    cost_xx
        Cost function in the incomparable space.
    """

    xy: Optional[ArrayLike]
    xx: Optional[ArrayLike]
    a: ArrayLike
    b: ArrayLike
    conditions: Optional[ArrayLike]
    cost_xy: OttCostFn_t
    cost_xx: OttCostFn_t

    @property
    def contains_linear(self) -> bool:
        """Whether the distribution contains data corresponding to the linear term."""
        return self.xy is not None

    @property
    def contains_quadratic(self) -> bool:
        """Whether the distribution contains data corresponding to the quadratic term."""
        return self.xx is not None

    @property
    def contains_condition(self) -> bool:
        """Whether the distribution contains data corresponding to the condition."""
        return self.conditions is not None

    @staticmethod
    def _extract_data(
        adata: AnnData,
        *,
        attr: Literal["X", "obs", "obsp", "obsm", "var", "varm", "layers", "uns"],
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

        if attr == "obs":
            data = np.asarray(data)[:, None]
        if sp.issparse(data):
            logger.warning(f"Densifying data in `{modifier}`")
            data = data.A
        if data.ndim != 2:
            raise ValueError(f"Expected `{modifier}` to have `2` dimensions, found `{data.ndim}`.")

        return data

    @staticmethod
    def _verify_input(
        xy_attr: Optional[Literal["X", "obsp", "obsm", "layers", "uns"]],
        xy_key: Optional[str],
        xx_attr: Optional[Literal["X", "obsp", "obsm", "layers", "uns"]],
        xx_key: Optional[str],
        conditions_attr: Optional[Literal["obs", "var", "obsm", "varm", "layers", "uns"]],
        conditions_key: Optional[str],
    ) -> Tuple[bool, bool, bool]:
        if (xy_attr is None and xy_key is not None) or (xy_attr is not None and xy_key is None):
            raise ValueError(r"Either both `xy_attr` and `xy_key` must be `None` or none of them.")
        if (xx_attr is None and xx_key is not None) or (xx_attr is not None and xx_key is None):
            raise ValueError(r"Either both `xy_attr` and `xy_key` must be `None` or none of them.")
        if (conditions_attr is None and conditions_key is not None) or (
            conditions_attr is not None and conditions_key is None
        ):
            raise ValueError(r"Either both `conditions_attr` and `conditions_key` must be `None` or none of them.")
        return xy_attr is not None, xx_attr is not None, conditions_attr is not None

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        a: ArrayLike,
        b: ArrayLike,
        xy_attr: Literal["X", "obsp", "obsm", "layers", "uns"] = None,
        xy_key: Optional[str] = None,
        xy_cost: CostFn_t = "sq_euclidean",
        xx_attr: Literal["X", "obsp", "obsm", "layers", "uns"] = None,
        xx_key: Optional[str] = None,
        xx_cost: CostFn_t = "sq_euclidean",
        conditions_attr: Optional[Literal["obs", "var", "obsm", "varm", "layers", "uns"]] = None,
        conditions_key: Optional[str] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> "DistributionContainer":
        """Create distribution container from :class:`~anndata.AnnData`.

        .. warning::
            Sparse arrays will be always densified.

        Parameters
        ----------
        adata
            Annotated data object.
        a
            Marginals when used as source distribution.
        b
            Marginals when used as target distribution.
        xy_attr
            Attribute of `adata` containing the data for the shared space.
        xy_key
            Key of `xy_attr` containing the data for the shared space.
        xy_cost
            Cost function when in the shared space.
        xx_attr
            Attribute of `adata` containing the data for the incomparable space.
        xx_key
            Key of `xx_attr` containing the data for the incomparable space.
        xx_cost
            Cost function in the incomparable space.
        conditions_attr
            Attribute of `adata` containing the conditions.
        conditions_key
            Key of `conditions_attr` containing the conditions.
        backend
            Backend to use.
        kwargs
            Keyword arguments to pass to the cost functions.

        Returns
        -------
        The distribution container.
        """
        contains_linear, contains_quadratic, contains_condition = cls._verify_input(
            xy_attr, xy_key, xx_attr, xx_key, conditions_attr, conditions_key
        )

        if contains_linear:
            xy_data = cls._extract_data(adata, attr=xy_attr, key=xy_key)
            xy_cost_fn = get_cost(xy_cost, backend=backend, **kwargs)
        else:
            xy_data = None
            xy_cost_fn = None

        if contains_quadratic:
            xx_data = cls._extract_data(adata, attr=xx_attr, key=xx_key)
            xx_cost_fn = get_cost(xx_cost, backend=backend, **kwargs)
        else:
            xx_data = None
            xx_cost_fn = None

        conditions_data = (
            cls._extract_data(adata, attr=conditions_attr, key=conditions_key) if contains_condition else None  # type: ignore[arg-type]  # noqa:E501
        )
        return cls(xy=xy_data, xx=xx_data, a=a, b=b, conditions=conditions_data, cost_xy=xy_cost_fn, cost_xx=xx_cost_fn)


class DistributionCollection(dict[K, DistributionContainer]):
    """Collection of distributions."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{list(self.keys())}"

    def __str__(self) -> str:
        return repr(self)

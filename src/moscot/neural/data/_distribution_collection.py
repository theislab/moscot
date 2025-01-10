from dataclasses import dataclass
from typing import Any, Hashable, Literal, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from anndata import AnnData

from moscot._logging import logger
from moscot._types import CostFn_t
from moscot.costs import get_cost

K = TypeVar("K", bound=Hashable)


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
    conditions
        Conditions for the distributions.
    cost_xy
        Cost function when in the shared space.
    cost_xx
        Cost function in the incomparable space.
    """

    xy: Optional[jax.Array]
    xx: Optional[jax.Array]
    conditions: Optional[jax.Array]
    cost_xy: Any
    cost_xx: Any

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

    @property
    def n_samples(self) -> int:
        """Number of samples in the distribution."""
        return self.xy.shape[0] if self.contains_linear else self.xx.shape[0]  # type: ignore[union-attr]

    @staticmethod
    def _extract_data(
        adata: AnnData,
        *,
        attr: Literal["X", "obs", "obsp", "obsm", "var", "varm", "layers", "uns"],
        key: Optional[str] = None,
    ) -> jax.Array:
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
            data = data.toarray()
        if data.ndim != 2:
            raise ValueError(f"Expected `{modifier}` to have `2` dimensions, found `{data.ndim}`.")

        return jnp.array(data)

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
        return cls(xy=xy_data, xx=xx_data, conditions=conditions_data, cost_xy=xy_cost_fn, cost_xx=xx_cost_fn)

    def __getitem__(
        self, idx: Union[int, slice, jnp.ndarray, jax.Array, list[Any], tuple[Any]]
    ) -> "DistributionContainer":
        """
        Return a new DistributionContainer where .xy, .xx, .conditions
        are sliced by `idx` (if they are not None).

        This allows usage like:
            new_container = distribution_container[train_ixs]
        """  # noqa: D205
        # TODO: Normally this is inefficient
        # But we first need to separate the slicing of training and validation data
        # Before creating this DistributionContainer!
        # Slice xy
        new_xy = self.xy[idx] if self.xy is not None else None

        # Slice xx
        new_xx = self.xx[idx] if self.xx is not None else None

        # Slice conditions
        new_conditions = self.conditions[idx] if self.conditions is not None else None

        # Reuse the same cost functions
        return DistributionContainer(
            xy=new_xy,
            xx=new_xx,
            conditions=new_conditions,
            cost_xy=self.cost_xy,
            cost_xx=self.cost_xx,
        )


class DistributionCollection(dict[K, DistributionContainer]):
    """Collection of distributions."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{list(self.keys())}"

    def __str__(self) -> str:
        return repr(self)

from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from ott.geometry.costs import CostFn
from ott.core.gromov_wasserstein import GWKlLoss, GWSqEuclLoss

from scott._base import BaseCostFn
from scott._mixins import GWLossMixin


# TODO(michalk8): necessary? if not, remove
@register_pytree_node_class
class WeightedCostFn(CostFn, BaseCostFn):
    """TODO."""

    def __init__(self, cost_a: CostFn, cost_b: CostFn, *, weight: float):
        if not (0 < weight < 1):
            raise ValueError("TODO")
        if not isinstance(cost_a, CostFn):
            raise TypeError("TODO.")
        if not isinstance(cost_b, CostFn):
            raise TypeError("TODO.")
        self._cost_a = cost_a
        self._cost_b = cost_b
        self._weight = weight

    def norm(self, x: jnp.array) -> jnp.array:
        """TODO."""
        return self.w * self._cost_a.norm(x) + (1 - self.w) * self._cost_b.norm(x)

    def pairwise(self, x: jnp.array, y: jnp.array) -> jnp.array:
        """TODO."""
        return self.w * self._cost_a.pairwise(x, y) + (1 - self.w) * self._cost_b.pairwise(x, y)

    @property
    def w(self) -> float:
        """TODO."""
        return self._weight

    # TODO(michalk8): correct?
    def tree_flatten(self):
        """TODO."""
        return (), (self._cost_a, self._cost_b, self.w)

    # TODO(michalk8): correct?
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """TODO."""
        del children
        return cls(aux_data[0], aux_data[1], weight=aux_data[2])


# TODO(michalk8): rename
class GWSeuqCostFn(GWLossMixin, GWSqEuclLoss):
    pass


class GWKlCostFn(GWLossMixin, GWKlLoss):
    pass

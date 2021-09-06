from abc import ABC
from typing import Any, Optional

from jax import numpy as jnp
from ott.tools.transport import Transport
from ott.geometry.geometry import Geometry

from scott._base import BaseCostFn


class GeomMixin:
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._geom: Optional[Geometry] = None
        self._f: Optional[jnp.ndarray] = None
        self._g: Optional[jnp.ndarray] = None

    @property
    def geometry(self) -> Geometry:
        return self._geom

    @property
    def matrix(self) -> Optional[jnp.ndarray]:
        if self.geometry is None:
            return None
        try:
            return self.geometry.transport_from_potentials(self._f, self._g)
        except ValueError:
            u = self.geometry.scaling_from_potential(self._f)
            v = self.geometry.scaling_from_potential(self._g)
            return self.geometry.transport_from_scalings(u, v)

    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> Optional[jnp.ndarray]:
        if self.geometry is None:
            return None
        # TODO(michalk8): correct axis?
        axis = 0 if forward else 1
        try:
            return self.geometry.apply_transport_from_potentials(self._f, self._g, inputs, axis=axis)
        except ValueError:
            u = self.geometry.scaling_from_potential(self._f)
            v = self.geometry.scaling_from_potential(self._g)
            return self.geometry.apply_transport_from_scalings(u, v, inputs, axis=axis)


class TransportMixin:
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._transport: Optional[Transport] = None

    @property
    def matrix(self) -> Optional[jnp.array]:
        return None if self._transport is None else self._transport.matrix

    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.array:
        # TODO(michalk8): correct axis?
        return self._transport.apply(inputs, axis=0 if forward else 1)


class GWLossMixin(BaseCostFn, ABC):
    # TODO(michalk8): verify correctness + shapes + add protocol
    def __call__(self, x: jnp.array, y: jnp.array):
        return self.fn_x(x) + self.fn_y(y) - self.left_x(x) * self.right_y(y)

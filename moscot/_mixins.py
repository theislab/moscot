from typing import Any, Optional

from jax import numpy as jnp
from ott.tools.transport import Transport
from ott.geometry.geometry import Geometry


class GeomMixin:
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._geom: Optional[Geometry] = None
        self._f: Optional[jnp.ndarray] = None
        self._g: Optional[jnp.ndarray] = None

    # TODO(michalk8): do we want to expose the geometry?
    @property
    def geometry(self) -> Geometry:
        """Underlying geometry."""
        return self._geom

    @property
    def matrix(self) -> jnp.ndarray:
        """Transport matrix."""
        # TODO(michalk8): improve message/ensure fitted (use sklearn)?
        if self.geometry is None:
            raise RuntimeError("Not fitted.")
        try:
            return self.geometry.transport_from_potentials(self._f, self._g)
        except ValueError:
            u = self.geometry.scaling_from_potential(self._f)
            v = self.geometry.scaling_from_potential(self._g)
            return self.geometry.transport_from_scalings(u, v)

    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
        """Transport mass."""
        if self.geometry is None:
            raise RuntimeError("Not fitted.")
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
    def converged(self) -> Optional[bool]:
        """`True` if the solver converged."""
        return None if self._transport is None else bool(self._transport.converged)

    @property
    def matrix(self) -> jnp.array:
        """Transport matrix."""
        if self._transport is None:
            raise RuntimeError("Not fitted.")
        return self._transport.matrix

    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.array:
        """Transport mass."""
        if self._transport is None:
            raise RuntimeError("Not fitted.")
        return self._transport.apply(inputs, axis=0 if forward else 1)


class SimpleMixin:
    # TODO(michalk8): polish this
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._matrix: Optional[jnp.ndarray] = None
        self._converged: Optional[bool] = None

    @property
    def converged(self) -> Optional[bool]:
        """`True` if the solver converged."""
        return self._converged

    @property
    def matrix(self) -> jnp.array:
        """Transport matrix."""
        # TODO(michalk8): unify interface (None or raise)
        if self._matrix is None:
            raise RuntimeError("Not fitted.")
        return self._matrix

    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.array:
        """Transport mass."""
        matrix = self.matrix.T if forward else self.matrix
        return matrix @ inputs

from abc import ABC, abstractmethod
from typing import Any, Optional

import ot

from ott.geometry import Geometry, PointCloud
import numpy as np

# TODO: Here we need to work on the backend registry, @Michalk8


class Geom(ABC):
    """
    This class wraps the cost matrix for different backends.
    self._geom should be a class allowing to be called by
        - self._geom(cost_matrix), where cost_matrix is np.ndarray
        - self._geom(X, Y, cost_function, online), where cost_function is applied to pairs of X and Y
        - self._geom(X, cost_function, online), where cost_function is applied to all pairs within X
    """

    def __init__(self, **kwargs):
        self._geom = None

    @abstractmethod
    @property
    def cost_matrix(self):
        pass

    @abstractmethod
    @property
    def shape(self):
        pass


class OTT_Geom(Geom):
    """
    Wrapper for ott.geometry.Geometry
    """

    def __init__(self, cost_matrix=None, x=None, **kwargs: Any):
        self._x = x
        self._cost_matrix = cost_matrix
        if cost_matrix is not None:
            self._geom = Geometry(cost_matrix)
        else:
            y = kwargs.pop("y", None)
            online = kwargs.pop("online", None)
            cost_fn = kwargs.pop("cost_fn", None)
            self._geom = PointCloud(x=x, y=y, online=online, cost_fn=cost_fn, **kwargs)

    def __getattr__(
        self, item
    ):  # https://stackoverflow.com/questions/65754399/conditional-inheritance-based-on-arguments-in-python
        return self._geom.__getattribute__(item)

    @property
    def cost_matrix(self):
        return self.cost_matrix

    @property
    def shape(self):
        return self.shape


class NP_Geom(Geom):
    """
    An example for a wrapper of a simple custom numpy backend using POT to calculate the distance
    """

    def __init__(
        self,
        cost_matrix: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        cost_fn: Optional[str] = "sqeuclidean",
        online: bool = False,
        **kwargs: Any,
    ):
        if online:
            raise ValueError("'NP_Geom' does not support online saving of the transport matrix.")
        if cost_matrix is not None:
            self._cost_matrix = cost_matrix
        elif x is not None:
            if y is not None:
                self._cost_matrix = ot.dist(x, y, metric=cost_fn, **kwargs)
            else:
                self._cost_matrix = ot.dist(x, x, metric=cost_fn, **kwargs)
        else:
            raise ValueError("cost_matrix or X must be given.")

    @property
    def cost_matrix(self):
        return self._cost_matrix

    @property
    def shape(self):
        return self._cost_matrix.shape

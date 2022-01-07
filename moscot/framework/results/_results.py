from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number

from abc import ABC, abstractmethod
import numpy as np

class OTResult(ABC):
    """ Abstract base class parenting classes to solve OT downstream functions. Can be manually provided"""
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def transport_matrix(self,
                         pre_transport_matrix: Any) -> np.ndarray:
        """
        Instantiates the transport matrix from pre_transport_matrix, which could be e.g. the cost matrix itself or a
        list of two potentials.
        """
        pass

    @staticmethod
    @abstractmethod
    def _push_forward(mass: Any,
                      matrix: Any) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def _pull_back(mass: Any,
                   matrix: Any) -> np.ndarray:
        pass



class MatrixOTResult(OTResult):
    """Class handling OT results whose transport matrix is saved as a matrix."""

    def __init__(self,
                 ) -> None:
        pass

    @staticmethod
    def transport_matrix(self,
                         pre_transport_matrix: np.ndarray) -> np.ndarray:
        return pre_transport_matrix

    @staticmethod
    def _push_forward(mass: np.ndarray,
                      matrix: np.ndarray) -> np.ndarray:
        return np.dot(mass, matrix)

    @staticmethod
    def _pull_back(mass: np.ndarray,
                   matrix: np.ndarray) -> np.ndarray:
        return np.transpose(np.dot(matrix, mass))


class PotentialOTResult(OTResult):
    """Class handling OT results which are based on potentials."""

    def __init__(self,
                 ) -> None:
        pass

    @staticmethod
    def transport_matrix(self,
                         pre_transport_matrix: List[np.ndarray, np.ndarray]) -> np.ndarray:
        return self._transport_matrix_from_potentials(pre_transport_matrix[0], pre_transport_matrix[1])

    @staticmethod
    def _transport_matrix_from_potentials(self,
                                          f: np.ndarray,
                                          g: np.ndarray) -> np.ndarray:
        pass  # TODO: @MUCDK implement

    @staticmethod
    def _push_forward(mass: np.ndarray,
                      potentials: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        pass #TODO: @MUCDK implement

    @staticmethod
    def _pull_back(mass: np.ndarray,
                   potentials: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        pass #TODO: @MUCDK implement
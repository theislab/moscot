from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number

from networkx import DiGraph
from abc import ABC, abstractmethod

from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss
import numpy as np
from anndata import AnnData

CostFn_t = Union[CostFn, GWLoss]



class BaseResult(ABC):
    """Base result for OT problems."""

    def __init__(self,
                 adata: AnnData,
                 key: str,
                 estimators_dict: Dict[Tuple, 'BaseProblem']) -> None:
        self._adata = adata
        self._key = key
        self._estimators_dict = estimators_dict


    def group_cells(self, key: Union[str, List[str]]) -> List:
        if isinstance(key, str):
            adata_groups = self._adata.obs[key].values
            assert key in adata_groups
            self._groups = list(adata_groups)
        else:
            self._groups = key

    @property
    def adata(self) -> AnnData:
        return self._adata

    @property
    def estimators_dict(self) -> Union[List['BaseProblem'], 'BaseProblem']:
        return self._estimators_dict

    @staticmethod
    def _push_forward(mass: jnp.ndarray,
                      matrix: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(mass, matrix)

    @staticmethod
    def _pull_back(mass: jnp.ndarray,
                   matrix: jnp.ndarray) -> jnp.ndarray:
        return jnp.transpose(jnp.dot(matrix, mass))


class OTResult(BaseResult):
    def __init__(self,
                 adata: AnnData,
                 key: str,
                 estimators_dict: Dict[Tuple, 'BaseProblem']) -> None:
        self.matrix_dict = {tup: getattr(estimator, "matrix") for tup, estimator in estimators_dict.items()}
        super().__init__(adata=adata, key=key, estimators_dict=estimators_dict)

    def plot_aggregate_transport(self, key: str, groups: List[str]):
        pass

    def push_forward_composed(self,
                              start: int,
                              end: int,
                              key_groups: Optional[str],
                              groups: Optional[List[str]]
                              ) -> jnp.ndarray:
        if start < end:
            raise ValueError("push_forward_composed() requires start < end.")


    def push_forward(self,
                     start: int, #TODO: check what type the key values should be
                     end: int,
                     key_groups: Optional[str] = None,
                     groups: Optional[List[str]] = None,
                     mass: Optional[np.ndarray] = None,
                     ) -> jnp.ndarray:

        if start < end:
            raise ValueError("push_forward() requires start < end.")
        if (start, end) not in self.matrix_dict.keys():
            raise ValueError("The transport matrix for the tuple {} has not been calculated. Try running 'push_forward_composed' instead".format((start, end)))
        matrix = self.matrix_dict[(start, end)]
        if mass is None:
            mass = self._prepare_transport(matrix, start, groups, key_groups)
        else:
            self._verify_mass(matrix, mass, 0)
        return self._push_forward(mass, matrix)

    def pull_back(self,
                  start: int, #TODO: check what type the key values should be
                  end: int,
                  key_groups: Optional[str] = None,
                  groups: Optional[List[str]] = None,
                  mass: Optional[np.ndarray] = None,
                  ) -> jnp.ndarray:
        if end < start:
            raise ValueError("pull_back() requires start > end.")
        if (end, start) not in self.matrix_dict.keys():
            raise ValueError("The transport matrix for the tuple {} has not been calculated. Try running 'pull_back_composed' instead".format((end, start)))
        matrix = self.matrix_dict[(end, start)]
        if mass is None:
            mass = self._prepare_transport(matrix, start, groups, key_groups)
        else:
            self._verify_mass(matrix, mass, 1)
        return self._pull_back(mass, matrix)

    @staticmethod
    def _verify_mass(matrix, mass, dimension):
        if len(mass.shape) != 1:
            raise ValueError("The dimension of provided mass must be 1.")
        if matrix.shape[dimension] == len(mass):
            raise ValueError("The dimensions of the matrix and the mass do not match.")

    def _prepare_transport(self,
                           matrix,
                           source,
                           groups,
                           key_groups):
        if ((groups == groups) + (key_groups == key_groups)) == 1:
            raise ValueError("Either both variables 'key_groups' and 'groups' must be provided or none of them.")
        if groups is not None:
            if key_groups not in self._adata.obs.columns:
                raise ValueError("key_groups {} not found in AnnData.obs.columns".format(key_groups))

            self._verify_cells_are_present(source, groups, key_groups)
            return self._make_mass_from_cells(source, groups, key_groups)
        else:
            dim_0 = matrix.shape[0]
            return jnp.full(dim_0, 1/dim_0)

    def _verify_cells_are_present(self,
                                  start: int,
                                  groups: List[str],
                                  key_groups: str) -> None:
        adata_groups = set(self._adata[self._adata.obs[self._key] == start][self._adata.obs[key_groups]].values)
        for group in groups:
            if group not in adata_groups:
                raise ValueError("Group {} is not present for considered data point {}".format(group, start))

    def _make_mass_from_cells(self,
                              start: int,
                              groups: List[str],
                              key_groups: str) -> jnp.ndarray:
        groups_at_start = self._adata[self._adata.obs[self._key] == start].obs[key_groups].values
        mass = jnp.zeros(len(groups_at_start))
        in_group = [group in groups for group in groups_at_start]
        n_cells = jnp.sum(in_group)
        mass[in_group] = 1/n_cells
        return mass



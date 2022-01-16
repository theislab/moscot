from abc import ABC
from typing import Any, List, Union, Optional

from ott.geometry.costs import CostFn
from ott.core.gromov_wasserstein import GWLoss
import numpy as np

from anndata import AnnData

from moscot.framework.results._results import LRResult, OTResult, MatrixOTResult, PotentialOTResult

CostFn_t = Union[CostFn, GWLoss]


class ResultMixin(ABC):
    """Mixin class for OT estimators handling downstream functions"""

    # TODO: @giovp, @michalk8, @MUCDK: Do we need all of these methods as they are for spatial problems?
    # TODO continued: probably not, hence move those not needed to TemporalResultMixin
    def __init__(
        self,
        adata: AnnData,
        otResult: Union[OTResult, "from_matrix", "from_potentials"],
    ) -> None:

        self._adata = adata  # @TODO: dicuss if we want this attribute. MUCDK: I like it because this way the adata
        # TODO: object only has to be passed on once as it is needed in multiple functions. Computational aspects? Please lmk
        if otResult == "from_matrix":
            self._otResult = MatrixOTResult
        elif otResult == "from_potentials":
            self._otResult = PotentialOTResult
        elif otResult == "low_rank":
            self._otResult = LRResult
        else:
            self._otResult = otResult

    def __getattr__(
        self, item
    ):  # https://stackoverflow.com/questions/65754399/conditional-inheritance-based-on-arguments-in-python
        return self._otResult.__getattribute__(item)

    @property
    def adata(self) -> AnnData:
        return self._adata

    def plot_aggregate_transport(self, key: str, groups: List[str]):
        pass

    def push_forward_composed(
        self, start: Any, end: Any, key_groups: Optional[str] = None, groups: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        if start > end:
            raise ValueError("push_forward_composed() requires start > end.")
        self._verify_sequence_exists(start, end)
        matrix = self.matrix_dict[(start, start + 1)]
        masses = [self._prepare_transport(matrix, start, groups, key_groups)]
        for i in range(start, end):
            masses += [self._push_forward(masses[-1], self.matrix_dict[(i, i + 1)])]
        return masses

    def pull_back_composed(
        self, start: Any, end: Any, key_groups: Optional[str] = None, groups: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        if start < end:
            raise ValueError("pull_back_composed() requires start < end.")
        self._verify_sequence_exists(end, start)
        matrix = self.matrix_dict[(start, start - 1)]
        masses = [self._prepare_transport(matrix, start, groups, key_groups)]
        for i in range(start, end, -1):
            masses += [self._pull_back(masses[-1], self.matrix_dict[(i - 1, i)])]
        return masses

    def push_forward(
        self,
        start: Any,  # TODO: check what type the key values should be
        end: Any,
        key_groups: Optional[str] = None,
        groups: Optional[List[str]] = None,
        mass: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:

        if start > end:
            raise ValueError("push_forward() requires start > end.")
        if (start, end) not in self.matrix_dict.keys():
            raise ValueError(
                "The transport matrix for the tuple {} has not been calculated. Try running 'push_forward_composed' instead".format(
                    (start, end)
                )
            )
        matrix = self.matrix_dict[(start, end)]
        if mass is None:
            mass = self._prepare_transport(matrix, start, groups, key_groups)
        else:
            self._verify_mass(matrix, mass, 0)
        return [mass, self._push_forward(mass, matrix)]

    def pull_back(
        self,
        start: Any,
        end: Any,
        key_groups: Optional[str] = None,
        groups: Optional[List[str]] = None,
        mass: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        if start < end:
            raise ValueError("pull_back() requires start < end.")
        if (end, start) not in self.matrix_dict.keys():
            raise ValueError(
                "The transport matrix for the tuple {} has not been calculated. Try running 'pull_back_composed' instead".format(
                    (end, start)
                )
            )
        matrix = self.matrix_dict[(end, start)]
        if mass is None:
            mass = self._prepare_transport(matrix, start, groups, key_groups)
        else:
            self._verify_mass(matrix, mass, 1)
        return [mass, self._pull_back(mass, matrix)]

    def group_cells(self, key: Union[str, List[str]]) -> List:
        if isinstance(key, str):
            adata_groups = self._adata.obs[key].values
            assert key in adata_groups
            self._groups = list(adata_groups)
        else:
            self._groups = key

    def _prepare_transport(self, matrix, source, groups, key_groups):
        if ((groups == groups) + (key_groups == key_groups)) == 1:
            raise ValueError("Either both variables 'key_groups' and 'groups' must be provided or none of them.")
        if groups is not None:
            if key_groups not in self._adata.obs.columns:
                raise ValueError(f"key_groups {key_groups} not found in AnnData.obs.columns")

            self._verify_cells_are_present(source, groups, key_groups)
            return self._make_mass_from_cells(source, groups, key_groups)
        else:
            dim_0 = matrix.shape[0]
            return np.full(dim_0, 1 / dim_0)

    def _verify_cells_are_present(self, start: Any, groups: List[str], key_groups: str) -> None:
        adata_groups = set(self._adata[self._adata.obs[self._key] == start][self._adata.obs[key_groups]].values)
        for group in groups:
            if group not in adata_groups:
                raise ValueError(f"Group {group} is not present for considered data point {start}")

    def _make_mass_from_cells(self, start: Any, groups: List[str], key_groups: str) -> np.ndarray:
        groups_at_start = self._adata[self._adata.obs[self._key] == start].obs[key_groups].values
        mass = np.zeros(len(groups_at_start))
        in_group = [group in groups for group in groups_at_start]
        n_cells = np.sum(in_group)
        mass[in_group] = 1 / n_cells
        return mass

    def _verify_sequence_exists(self, early: int, late: int) -> None:
        matrix_keys = self.matrix_dict.keys()
        for i in range(early, late):
            if (i, i + 1) not in matrix_keys:
                raise ValueError(f"No transport matrix was calculated for {(i, i + 1)}")

    @staticmethod
    def _verify_mass(matrix, mass, dimension):
        if len(mass.shape) != 1:
            raise ValueError("The dimension of provided mass must be 1.")
        if matrix.shape[dimension] == len(mass):
            raise ValueError("The dimensions of the matrix and the mass do not match.")


class TemporalResultMixin(ResultMixin):
    """This class handles time-specific downstream functions
    TODO: check with @giovp which methods in ResultMixin he needs to change / which are not sufficiently generic
    TODO continued: and add them to this class"""


class SpatialResultMixin(ResultMixin):
    """This class handles downstream functions of spatial problems"""

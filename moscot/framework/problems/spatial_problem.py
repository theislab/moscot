from typing import Any, Union

from ott.geometry.costs import CostFn
from ott.core.gromov_wasserstein import GWLoss

from anndata import AnnData

from moscot.framework.utils.custom_costs import Leaf_distance
from moscot.framework.problems.BaseProblem import BaseProblem
from moscot.framework.results._result_mixins import SpatialResultMixin

CostFn_t = Union[CostFn, GWLoss]
CostFn_tree = Union[Leaf_distance]
CostFn_general = Union[CostFn_t, CostFn_tree]
Scales = Union["mean", "meadian", "max"]


class SpatialProblem(BaseProblem, SpatialResultMixin):
    """This estimator handles all OT problems related to spatial sc data"""


class SpatialAlignmentEstimator(SpatialProblem):
    """
    This estimator ...
    """

    def __init__(
        self,
        adata: AnnData,
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, **kwargs)


class SpatialMappingEstimator(SpatialProblem):
    """
    This estimator ...
    """

    def __init__(
        self,
        adata: AnnData,
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, **kwargs)

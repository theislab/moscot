from enum import Enum
from typing import Optional
from dataclasses import dataclass

import numpy.typing as npt


class Tag(str, Enum):
    COST_MATRIX = "cost"
    KERNEL = "kernel"
    POINT_CLOUD = "point_cloud"
    GRID = "grid"
    COMPUTE_COST = "compute_cost"


@dataclass(frozen=True, repr=True)
class TaggedArray:
    # passed to solver._prepare_input
    data: npt.ArrayLike
    tag: Tag = Tag.POINT_CLOUD  # TODO(michalk8): in post_init, do check if it's correct type
    loss: Optional[
        str
    ] = None  # if cost matrix is in data we don't need loss. Easier to read the code if loss is then set to None

    @property
    def is_cost_matrix(self) -> bool:
        return self.tag == Tag.COST_MATRIX

    @property
    def is_kernel(self) -> bool:
        return self.tag == Tag.KERNEL

    @property
    def is_point_cloud(self) -> bool:
        return self.tag == Tag.POINT_CLOUD

    @property
    def is_grid(self) -> bool:
        return self.tag == Tag.GRID

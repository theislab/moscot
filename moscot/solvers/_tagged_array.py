from enum import Enum
from typing import Optional
from dataclasses import dataclass

import numpy.typing as npt


class Tag(str, Enum):
    """Tag of :class:`moscot.solvers._tagged_array.TaggedArray`."""

    COST_MATRIX = "cost"
    KERNEL = "kernel"
    POINT_CLOUD = "point_cloud"
    GRID = "grid"


@dataclass(frozen=True, repr=True)
class TaggedArray:
    """Tagged Array."""

    # passed to solver._prepare_input
    data: npt.ArrayLike
    tag: Tag = Tag.POINT_CLOUD  # TODO(michalk8): in post_init, do check if it's correct type
    loss: Optional[
        str
    ] = None  # if cost matrix is in data we don't need loss. Easier to read the code if loss is then set to None

    @property
    def is_cost_matrix(self) -> bool:
        """Return `True` if :class:`moscot.solvers._tagged_array.TaggedArray` contains a cost matrix."""
        return self.tag == Tag.COST_MATRIX

    @property
    def is_kernel(self) -> bool:
        """Return `True` if :class:`moscot.solvers._tagged_array.TaggedArray` contains a kernel."""
        return self.tag == Tag.KERNEL

    @property
    def is_point_cloud(self) -> bool:
        """Return `True` if :class:`moscot.solvers._tagged_array.TaggedArray` contains a point cloud."""
        return self.tag == Tag.POINT_CLOUD

    @property
    def is_grid(self) -> bool:
        """Return `True` if :class:`moscot.solvers._tagged_array.TaggedArray` contains a grid."""
        return self.tag == Tag.GRID

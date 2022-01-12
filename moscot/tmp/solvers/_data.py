from enum import auto, Enum
from dataclasses import dataclass
from typing import Union
import numpy.typing as npt

Loss = Union[str, ]

class Tag(Enum):
    COST_MATRIX = auto()
    KERNEL = auto()
    POINT_CLOUD = auto()
    GRID = auto()


@dataclass(frozen=True, repr=True)
class TaggedArray:
    # passed to solver._prepare_input
    data: npt.ArrayLike
    tag: Tag = Tag.POINT_CLOUD  # TODO(michalk8): in post_init, do check if it's correct type
    loss: Loss = "sqeucl"

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

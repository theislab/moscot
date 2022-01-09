from enum import auto, Enum
from dataclasses import dataclass

import numpy.typing as npt


class Tag(Enum):
    COST = auto()
    KERNEL = auto()
    POINT_CLOUD = auto()
    GRID = auto()


@dataclass(frozen=True, repr=True)
class TaggedArray:
    # passed to solver._prepare_input
    data: npt.ArrayLike
    tag: Tag = Tag.POINT_CLOUD  # TODO(michalk8): in post_init, do check if it's correct type
    # TODO(michalk8): add loss here so that OTT can construct geometry

    @property
    def is_cost(self) -> bool:
        return self.tag == Tag.COST

    @property
    def is_kernel(self) -> bool:
        return self.tag == Tag.KERNEL

    @property
    def is_point_cloud(self) -> bool:
        return self.tag == Tag.POINT_CLOUD

    @property
    def is_grid(self) -> bool:
        return self.tag == Tag.GRID

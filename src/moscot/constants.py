import enum

__all__ = [
    "ProblemStage",
    "ProblemKind",
    "Tag",
    "Policy",
    "AlignmentMode",
    "CorrMethod",
    "AggregationMode",
    "PlottingKeys",
    "PlottingDefaults",
    "CorrTestMethod",
]


class StrEnum(str, enum.Enum):
    pass


@enum.unique
class ProblemStage(StrEnum):
    INITIALIZED = "initialized"
    PREPARED = "prepared"
    SOLVED = "solved"


@enum.unique
class ProblemKind(StrEnum):
    """Type of optimal transport problems."""

    UNKNOWN = "unknown"
    LINEAR = "linear"
    QUAD = "quadratic"


@enum.unique
class Tag(str, enum.Enum):
    """Tag used to interpret array-like data in :class:`moscot.solvers.TaggedArray`."""

    # TODO(michalk8): document rest of the classes
    COST_MATRIX = "cost_matrix"  #: Cost matrix.
    KERNEL = "kernel"  #: Kernel matrix.
    POINT_CLOUD = "point_cloud"  #: Point cloud.


@enum.unique
class Policy(StrEnum):
    SEQUENTIAL = "sequential"
    STAR = "star"
    EXTERNAL_STAR = "external_star"
    DUMMY = "dummy"
    TRIU = "triu"
    TRIL = "tril"
    EXPLICIT = "explicit"


@enum.unique
class AlignmentMode(StrEnum):
    AFFINE = "affine"
    WARP = "warp"


@enum.unique
class CorrMethod(StrEnum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"


@enum.unique
class AggregationMode(StrEnum):
    ANNOTATION = "annotation"
    CELL = "cell"


@enum.unique
class PlottingKeys(StrEnum):  # sets the adata.uns[AdataKeys.UNS][value] values
    CELL_TRANSITION = "cell_transition"
    SANKEY = "sankey"
    PUSH = "push"
    PULL = "pull"


@enum.unique
class PlottingDefaults(StrEnum):  # sets the adata.uns[AdataKeys.UNS][value] values
    CELL_TRANSITION = "cell_transition"
    SANKEY = "sankey"
    PUSH = "push"
    PULL = "pull"


@enum.unique
class CorrTestMethod(StrEnum):
    FISCHER = "fischer"
    PERM_TEST = "perm_test"

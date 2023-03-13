import enum

__all__ = [
    "ProblemStage",
    "ScaleCost",
    "Policy",
    "AlignmentMode",
    "CorrMethod",
    "AggregationMode",
    "AdataKeys",
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
class ScaleCost(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX_COST = "max_cost"
    MAX_BOUND = "max_bound"
    MAX_NORM = "max_norm"


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


# TODO(MUCKD): refactor, no need for enum
@enum.unique
class AdataKeys(StrEnum):  # sets default keys for adata attributes
    UNS = "moscot_results"


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

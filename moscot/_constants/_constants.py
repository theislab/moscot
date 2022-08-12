from enum import unique

from moscot._constants._enum import ModeEnum


@unique
class ScaleCost(ModeEnum):
    MEAN = "mean"
    MAX = "max"
    MEDIAN = "median"
    MAX_COST = "max_cost"
    MAX_BOUND = "max_bound"
    MAX_NORM = "max_norm"


@unique
class Policy(ModeEnum):
    SEQUENTIAL = "sequential"
    STAR = "star"
    EXTERNAL_STAR = "external_star"
    TRIU = "triu"
    TRIL = "tril"
    EXPLICIT = "explicit"


@unique
class AlignmentMode(ModeEnum):
    AFFINE = "affine"
    WARP = "warp"


@unique
class CorrMethod(ModeEnum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"


@unique
class AggregationMode(ModeEnum):
    ANNOTATION = "annotation"
    CELL = "cell"


@unique
class AdataKeys(ModeEnum):  # sets default keys for adata attributes
    UNS = "moscot_results"


@unique
class PlottingKeys(ModeEnum):  # sets the adata.uns[AdataKeys.UNS][value] values
    CELL_TRANSITION = "cell_transition"
    SANKEY = "sankey"


@unique
class PlottingDefaults(ModeEnum):  # sets the adata.uns[AdataKeys.UNS][value] values
    CELL_TRANSITION = "cell_transition"
    SANKEY = "sankey"

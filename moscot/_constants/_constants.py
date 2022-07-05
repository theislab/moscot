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
class ReferenceNaming(ModeEnum):
    SRC_NAME = "src"
    TGT_NAME = "ref"

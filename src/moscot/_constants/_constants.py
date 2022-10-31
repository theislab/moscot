from enum import unique
from typing import Any, Optional

from anndata import AnnData

from moscot._constants._enum import ModeEnum


@unique
class ProblemStage(ModeEnum):
    INITIALIZED = "initialized"
    PREPARED = "prepared"
    SOLVED = "solved"


@unique
class ScaleCost(ModeEnum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX_COST = "max_cost"
    MAX_BOUND = "max_bound"
    MAX_NORM = "max_norm"


@unique
class Policy(ModeEnum):
    SEQUENTIAL = "sequential"
    STAR = "star"
    EXTERNAL_STAR = "external_star"
    DUMMY = "dummy"
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


# TODO(MUCKD): refactor, no need for enum
@unique
class AdataKeys(ModeEnum):  # sets default keys for adata attributes
    UNS = "moscot_results"


@unique
class PlottingKeys(ModeEnum):  # sets the adata.uns[AdataKeys.UNS][value] values
    CELL_TRANSITION = "cell_transition"
    SANKEY = "sankey"
    PUSH = "push"
    PULL = "pull"


@unique
class PlottingDefaults(ModeEnum):  # sets the adata.uns[AdataKeys.UNS][value] values
    CELL_TRANSITION = "cell_transition"
    SANKEY = "sankey"
    PUSH = "push"
    PULL = "pull"


class Key:
    class uns:
        @classmethod
        def set_plotting_vars(
            cls,
            adata: AnnData,
            uns_key: str,
            pl_func_key: Optional[str] = None,
            key: Optional[str] = None,
            value: Optional[Any] = None,
            override: bool = True,
        ) -> None:
            adata.uns.setdefault(uns_key, {})
            if pl_func_key is not None:
                adata.uns[uns_key].setdefault(pl_func_key, {})
            if key is not None:
                if not override and key in adata.uns[uns_key][pl_func_key]:
                    raise KeyError(
                        f"Data in `adata.uns[{uns_key!r}][{pl_func_key!r}][{key!r}]` "
                        f"already exists, use `override=True`."
                    )
                adata.uns[uns_key][pl_func_key][key] = value

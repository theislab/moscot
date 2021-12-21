from typing import Literal, Union

# TODO: check where we want to save stuff like this to prevent circular imports
strategies_MatchingEstimator = Union[Literal["pairwise"], Literal["sequential"]]
from typing import Any

from moscot.problems.base import AnalysisMixin  # type: ignore[attr-defined]
from moscot.problems.base._compound_problem import B, K

__all__ = ["IntegrationMixin"]


class IntegrationMixin(AnalysisMixin[K, B]):
    """Integration analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

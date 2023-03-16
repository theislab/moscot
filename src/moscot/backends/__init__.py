from typing import Tuple

from moscot.backends import ott
from moscot.backends.utils import get_solver, register_solver

__all__ = ["ott", "get_solver", "register_solver", "get_available_backends"]


def get_available_backends() -> Tuple[str, ...]:
    """TODO."""
    return ("ott",)

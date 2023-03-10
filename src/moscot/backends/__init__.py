from typing import Tuple

from . import ott

# TODO(michalk8): consider importing `ott` backend and/or registering it
from .utils import get_solver, register_solver


def get_available_backends() -> Tuple[str, ...]:
    """TODO."""
    return ("ott",)

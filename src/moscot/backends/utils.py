from typing import Any, Union, Literal, Optional, TYPE_CHECKING

from moscot.base import solver  # TODO(michalk8): move to problem
from moscot.utils import registry

if TYPE_CHECKING:
    from . import ott

__all__ = ["get_solver", "register_solver"]


_REGISTRY = registry.Registry(fallback="ott")


def get_solver(problem_kind: solver.ProblemKind, *, backend: Optional[str] = None, **kwargs: Any) -> Any:
    """TODO."""
    if backend is None:
        backend = _REGISTRY.fallback
    if backend not in _REGISTRY:
        raise ValueError(f"Backend `{backend!r}` is not available.")
    return _REGISTRY[backend](problem_kind, **kwargs)


def register_solver(backend: str) -> Any:
    """TODO."""
    return _REGISTRY.register(backend)


@register_solver("ott")
def _(problem_kind: Literal["linear", "quadratic"], **kwargs: Any) -> Union["ott.SinkhornSolver", "ott.GWSolver"]:
    from . import ott

    if problem_kind == "linear":
        return ott.SinkhornSolver(**kwargs)
    if problem_kind == "quadratic":
        return ott.GWSolver(**kwargs)
    raise NotImplementedError(f"Unable to create solver for `{problem_kind!r}` problem.")

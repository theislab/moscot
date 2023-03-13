from typing import TYPE_CHECKING, Any, Literal, Union

from moscot._types import ProblemKind_t
from moscot.utils import registry

if TYPE_CHECKING:
    from . import ott

__all__ = ["get_solver", "register_solver"]


_REGISTRY = registry.Registry()


def get_solver(problem_kind: ProblemKind_t, *, backend: str = "ott", **kwargs: Any) -> Any:
    """TODO."""
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

from typing import TYPE_CHECKING, Any, Literal, Tuple, Type, Union

from moscot import _registry
from moscot._types import ProblemKind_t

if TYPE_CHECKING:
    from moscot.backends import ott

__all__ = ["get_solver", "register_solver", "get_available_backends"]


_REGISTRY = _registry.Registry()


def get_solver(problem_kind: ProblemKind_t, *, backend: str = "ott", return_class: bool = False, **kwargs: Any) -> Any:
    """TODO."""
    if backend not in _REGISTRY:
        raise ValueError(f"Backend `{backend!r}` is not available.")
    solver_class = _REGISTRY[backend](problem_kind)
    return solver_class if return_class else solver_class(**kwargs)


def register_solver(backend: str) -> Any:
    """TODO."""
    return _REGISTRY.register(backend)


@register_solver("ott")
def _(problem_kind: Literal["linear", "quadratic"]) -> Union[Type["ott.SinkhornSolver"], Type["ott.GWSolver"]]:
    from moscot.backends import ott

    if problem_kind == "linear":
        return ott.SinkhornSolver
    if problem_kind == "quadratic":
        return ott.GWSolver
    raise NotImplementedError(f"Unable to create solver for `{problem_kind!r}` problem.")


def get_available_backends() -> Tuple[str, ...]:
    """TODO."""
    return tuple(backend for backend in _REGISTRY)

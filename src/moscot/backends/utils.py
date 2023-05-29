from typing import TYPE_CHECKING, Any, Literal, Tuple, Union

from moscot import _registry
from moscot._types import ProblemKind_t

if TYPE_CHECKING:
    from moscot.backends import ott

__all__ = ["get_solver", "register_solver", "get_available_backends"]


_REGISTRY = _registry.Registry()


def get_solver(problem_kind: ProblemKind_t, *, backend: str = "ott", **kwargs: Any) -> Any:
    """TODO."""
    if backend not in _REGISTRY:
        raise ValueError(f"Backend `{backend!r}` is not available.")
    return _REGISTRY[backend](problem_kind, **kwargs)


def register_solver(backend: str) -> Any:
    """TODO."""
    return _REGISTRY.register(backend)


@register_solver("ott")
def _(
    problem_kind: Literal["linear", "quadratic"], neural: Union[bool, Literal["cond"]] = False, **kwargs: Any
) -> Union["ott.SinkhornSolver", "ott.GWSolver", "ott.NeuralSolver", "ott.CondNeuralSolver"]:
    from moscot.backends import ott

    if problem_kind == "linear":
        if neural:
            if neural == "cond":
                return ott.CondNeuralSolver(**kwargs)
            return ott.NeuralSolver(**kwargs)
        return ott.SinkhornSolver(**kwargs)
    if problem_kind == "quadratic":
        return ott.GWSolver(**kwargs)
    raise NotImplementedError(f"Unable to create solver for `{problem_kind!r}` problem.")


def get_available_backends() -> Tuple[str, ...]:
    """TODO."""
    return tuple(backend for backend in _REGISTRY)

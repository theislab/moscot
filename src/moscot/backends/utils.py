from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Tuple, Union

from moscot import _registry
from moscot._types import ProblemKind_t

if TYPE_CHECKING:
    from moscot.backends import ott

__all__ = ["get_solver", "register_solver", "get_available_backends"]

register_solver_t = Callable[
    [Literal["linear", "quadratic"], Optional[Literal["GENOTLinSolver"]]],
    Union["ott.SinkhornSolver", "ott.GWSolver", "ott.GENOTLinSolver"],
]


_REGISTRY = _registry.Registry()


def get_solver(problem_kind: ProblemKind_t, *, backend: str = "ott", return_class: bool = False, **kwargs: Any) -> Any:
    """TODO."""
    if backend not in _REGISTRY:
        raise ValueError(f"Backend `{backend!r}` is not available.")
    solver_class = _REGISTRY[backend](problem_kind, solver_name=kwargs.pop("solver_name", None))
    return solver_class if return_class else solver_class(**kwargs)


def register_solver(
    backend: str,
) -> Union["ott.SinkhornSolver", "ott.GWSolver", "ott.GENOTLinSolver"]:
    """Register a solver for a specific backend.

    Parameters
    ----------
    backend
        Name of the backend.

    Returns
    -------
    The decorated function which returns the type of the solver.
    """
    return _REGISTRY.register(backend)  # type: ignore[return-value]


# TODO(@MUCDK) fix mypy error
@register_solver("ott")  # type: ignore[arg-type]
def _(
    problem_kind: Literal["linear", "quadratic"],
    solver_name: Optional[Literal["GENOTLinSolver"]] = None,
) -> Union["ott.SinkhornSolver", "ott.GWSolver", "ott.GENOTLinSolver"]:
    from moscot.backends import ott

    if problem_kind == "linear":
        if solver_name == "GENOTLinSolver":
            return ott.GENOTLinSolver  # type: ignore[return-value]
        if solver_name is None:
            return ott.SinkhornSolver  # type: ignore[return-value]
    if problem_kind == "quadratic":
        return ott.GWSolver  # type: ignore[return-value]
    raise NotImplementedError(f"Unable to create solver for `{problem_kind!r}`, {solver_name} problem.")


def get_available_backends() -> Tuple[str, ...]:
    """Return all available backends."""
    return tuple(backend for backend in _REGISTRY)

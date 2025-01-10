from typing import TYPE_CHECKING, Any, Literal, Tuple, TypeVar, Union

from moscot import _registry
from moscot._types import ProblemKind_t

if TYPE_CHECKING:
    from moscot.backends import ott
    from moscot.neural.backends import neural_ott


__all__ = ["get_solver", "register_solver", "get_available_backends"]


_REGISTRY = _registry.Registry()

GWSolver = TypeVar("GWSolver", bound="ott.GWSolver")
SinkhornSolver = TypeVar("SinkhornSolver", bound="ott.SinkhornSolver")
GENOTSolver = TypeVar("GENOTSolver", bound="neural_ott.GENOTSolver")


def get_solver(problem_kind: ProblemKind_t, *, backend: str = "ott", return_class: bool = False, **kwargs: Any) -> Any:
    """TODO."""
    if backend not in _REGISTRY:
        raise ValueError(f"Backend `{backend!r}` is not available.")
    solver_class = _REGISTRY[backend](problem_kind, solver_name=kwargs.pop("solver_name", None))
    return solver_class if return_class else solver_class(**kwargs)


def register_solver(
    backend: str,
) -> Union[SinkhornSolver, GWSolver, GENOTSolver]:
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


@register_solver("ott")  # type: ignore[misc]
def create_ott_solver(
    problem_kind: Literal["linear", "quadratic"],
    solver_name: Any = None,
) -> Union[SinkhornSolver, GWSolver]:
    from moscot.backends import ott

    if problem_kind == "linear":
        return ott.SinkhornSolver  # type: ignore[return-value]
    if problem_kind == "quadratic":
        return ott.GWSolver  # type: ignore[return-value]
    raise NotImplementedError(f"Unable to create solver for `{problem_kind!r}`, {solver_name} problem.")


@register_solver("neural_ott")
def create_neural_ott_solver(
    problem_kind: Literal["linear", "quadratic"],
    solver_name: Any = None,
) -> GENOTSolver:
    from moscot.neural.backends import neural_ott

    if solver_name == "GENOTSolver":
        return neural_ott.GENOTSolver

    raise NotImplementedError(f"Unable to create solver for `{problem_kind!r}`, {solver_name} problem.")


def get_available_backends() -> Tuple[str, ...]:
    """Return all available backends."""
    return tuple(backend for backend in _REGISTRY)

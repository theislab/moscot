from typing import Any, Tuple, TypeVar, Callable, Optional
import enum

__all__ = ["register_backend", "get_backends", "get_solver"]


class ProblemKind(enum.Enum):
    """Optimal transport problems."""

    UNKNOWN = "unknown"
    LINEAR = "linear"
    QUAD = "quadratic"
    NEURAL = "neural"


F = TypeVar("F", bound=Callable[[ProblemKind, Any], Any])


class Registry:  # noqa: D101
    """TODO."""

    def __init__(self):
        self._registry = {}

    def register_backend(self, name: str) -> Callable[[F], F]:
        """TODO."""

        def decorator(func: F) -> F:
            self._registry[name] = func
            return func

        return decorator

    def available_backends(self) -> Tuple[str, ...]:
        """TODO."""
        return tuple(self._registry.keys())

    @property
    def default_backend(self) -> str:
        """TODO."""
        return "ott"

    def __contains__(self, backend: str) -> bool:
        return backend in self._registry

    def __getitem__(self, backend: str):
        return self._registry[backend]

    def __repr__(self) -> str:
        return repr(self._registry)

    def __str__(self) -> str:
        return str(self._registry)


def get_solver(problem_kind: ProblemKind, *, backend: Optional[str] = None, **kwargs: Any) -> Any:
    """TODO."""
    if backend is None:
        backend = _REGISTRY.default_backend
    if backend not in _REGISTRY:
        raise ValueError(f"Backend `{backend!r}` is not available.")
    return _REGISTRY[backend](problem_kind, **kwargs)


def get_backends() -> Tuple[str, ...]:
    """TODO."""
    return _REGISTRY.available_backends()


def register_backend(backend: str) -> Any:
    """TODO."""
    return _REGISTRY.register_backend(backend)


_REGISTRY = Registry()

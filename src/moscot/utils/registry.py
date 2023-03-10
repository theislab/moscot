from typing import Any, Tuple, TypeVar, Callable, Optional

__all__ = ["Registry"]


F = TypeVar("F", bound=Callable[..., Any])


class Registry:  # noqa: D101
    """TODO."""

    def __init__(self, fallback: Optional[str] = None):
        self._registry = {}  # type: ignore[var-annotated]
        self._fallback = fallback

    def register(self, name: str) -> Callable[[F], F]:
        """TODO."""

        def decorator(func: F) -> F:
            self._registry[name] = func
            return func

        return decorator

    def keys(self) -> Tuple[str, ...]:
        """TODO."""
        return tuple(self._registry.keys())

    @property
    def fallback(self) -> str:
        """TODO."""
        if self._fallback is None:
            raise ValueError("TODO")
        return self._fallback

    def __contains__(self, backend: str) -> bool:
        return backend in self._registry

    def __getitem__(self, backend: Optional[str]):
        return self._registry[backend]

    def __repr__(self) -> str:
        return repr(self._registry)

    def __str__(self) -> str:
        return str(self._registry)

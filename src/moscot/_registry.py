from typing import Any, Callable, Iterator, Optional, TypeVar

__all__ = ["Registry"]


F = TypeVar("F", bound=Callable[..., Any])


class Registry:  # noqa: D101
    """TODO."""

    def __init__(self):
        self._registry = {}

    def register(self, name: str) -> Callable[[F], F]:
        """TODO."""

        def decorator(func: F) -> F:
            self._registry[name] = func
            return func

        return decorator

    def __iter__(self) -> Iterator[str]:
        yield from self._registry.keys()

    def __contains__(self, backend: str) -> bool:
        return backend in self._registry

    def __getitem__(self, backend: Optional[str]):
        return self._registry[backend]

    def __len__(self):
        return len(self._registry)

    def __repr__(self) -> str:
        return repr(self._registry)

    def __str__(self) -> str:
        return str(self._registry)

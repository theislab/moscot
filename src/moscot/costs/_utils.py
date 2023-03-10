from typing import Any, Optional, Tuple

from moscot.utils import registry

__all__ = ["get_cost", "get_available_costs", "register_cost"]


_REGISTRY = registry.Registry()


def get_cost(name: str, *, backend: Optional[str] = None, **kwargs: Any) -> Any:
    """TODO."""
    if backend is not None:
        name = f"{backend}-{name}"
    if name not in _REGISTRY:
        raise ValueError(f"Cost `{name!r}` is not available.")
    return _REGISTRY[name](**kwargs)


def get_available_costs(backend: Optional[str] = None) -> Tuple[str, ...]:
    """TODO."""
    costs = _REGISTRY.keys()
    if backend is None:
        # TODO(michalk8): remove backend prefixes?
        return tuple(costs)
    prefix = f"{backend}-"
    return tuple(c.removeprefix(prefix) for c in costs if c.startswith(prefix))


def register_cost(name: str, *, backend: Optional[str] = None) -> Any:
    """TODO."""
    if backend is not None:
        name = f"{backend}-{name}"
    return _REGISTRY.register(name)

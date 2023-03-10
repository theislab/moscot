from typing import Any, Tuple

from moscot.utils import registry

__all__ = ["get_cost", "get_available_costs", "register_cost"]


_REGISTRY = registry.Registry()


def get_cost(name: str, **kwargs: Any) -> Any:
    """TODO."""
    if name not in _REGISTRY:
        raise ValueError(f"Cost `{name!r}` is not available.")
    return _REGISTRY[name](**kwargs)


def get_available_costs() -> Tuple[str, ...]:
    """TODO."""
    return _REGISTRY.keys()


def register_cost(name: str) -> Any:
    """TODO."""
    return _REGISTRY.register(name)

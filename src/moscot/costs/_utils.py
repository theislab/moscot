import collections
from typing import Any, Dict, List, Optional, Tuple

from moscot import _registry

__all__ = ["get_cost", "get_available_costs", "register_cost"]


_REGISTRY = _registry.Registry()
_SEP = "-"


def get_cost(name: str, *, backend: str = "moscot", **kwargs: Any) -> Any:
    """Get cost function for a specific backend."""
    key = f"{backend}{_SEP}{name}"
    if key not in _REGISTRY:
        raise ValueError(f"Cost `{name!r}` is not available for backend `{backend!r}`.")
    return _REGISTRY[key](**kwargs)


def get_available_costs(backend: Optional[str] = None) -> Dict[str, Tuple[str, ...]]:
    """Return available costs.

    Parameters
    ----------
    backend
        Select cost specific to a backend. If :obj:`None`, return the costs for each backend.

    Returns
    -------
    Dictionary with keys as backend names and values as registered cost functions.
    """
    groups: Dict[str, List[str]] = collections.defaultdict(list)
    for key in _REGISTRY:
        back, *name = key.split(_SEP)
        groups[back].append(_SEP.join(name))

    if backend is None:
        return {k: tuple(v) for k, v in groups.items()}
    if backend not in groups:
        raise KeyError(f"No backend named `{backend!r}`.")

    return {backend: tuple(groups[backend])}


def register_cost(name: str, *, backend: str) -> Any:
    """Register cost function for a specific backend."""
    return _REGISTRY.register(f"{backend}{_SEP}{name}")

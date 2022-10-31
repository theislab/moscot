from typing import Tuple, Optional

from moscot._constants._enum import ModeEnum

__all__ = ["get_backend", "get_available_backends", "get_default_backend"]


class Backend(ModeEnum):
    OTT = "ott"


def get_backend(backend: Optional[str] = None) -> str:
    """Request a backend.

    Parameters
    ----------
    backend
        Which backend to use, see :func:`moscot.backends.get_available_backends`.
        If `None`, use :func:`moscot.backends.get_default_backend`.

    Returns
    -------
    The requested backend.
    """
    if backend is None:
        return get_default_backend()
    return Backend(backend).value


def get_available_backends() -> Tuple[str, ...]:
    """Get available backends."""
    return tuple(Backend)


def get_default_backend() -> str:
    """Get the default backend."""
    return Backend.OTT.value

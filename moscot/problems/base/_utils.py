from types import MappingProxyType
from typing import Any, Dict, Type, Callable, Optional
from functools import partial, update_wrapper

__all__ = ["attributedispatch"]

Callback = Callable[..., Any]


def attributedispatch(func: Optional[Callback] = None, attr: Optional[str] = None) -> Callback:
    """Dispatch a function based on the first value."""

    def dispatch(value: Type[Any]) -> Callback:
        for typ in value.mro():
            if typ in registry:
                return registry[typ]
        return func  # type: ignore[return-value]

    def register(value: Type[Any], func: Optional[Callback] = None) -> Callback:
        if func is None:
            return lambda f: register(value, f)
        registry[value] = func
        return func

    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        typ = type(getattr(instance, str(attr)))
        return dispatch(typ)(instance, *args, **kwargs)

    if func is None:
        return partial(attributedispatch, attr=attr)

    registry: Dict[Type[Any], Callback] = {}
    wrapper.register = register  # type: ignore[attr-defined]
    wrapper.dispatch = dispatch  # type: ignore[attr-defined]
    wrapper.registry = MappingProxyType(registry)  # type: ignore[attr-defined]
    update_wrapper(wrapper, func)

    return wrapper

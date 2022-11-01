from typing import Any, Callable, TYPE_CHECKING
from textwrap import dedent


def inject_docs(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(obj: Any) -> Any:
        if TYPE_CHECKING:
            assert isinstance(obj.__doc__, str)
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj: Any) -> Any:
        obj.__doc__ = dedent(kwargs["__doc__"])
        return obj

    if isinstance(kwargs.get("__doc__", None), str) and len(kwargs) == 1:
        return decorator2

    return decorator

from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from typing import Any, Dict, Type, Tuple, Callable
from functools import wraps


class PrettyEnum(Enum):
    """Enum with a modified :meth:`__str__` and :meth:`__repr__`."""

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)


def _pretty_raise_enum(cls: Type["ErrorFormatterABC"], func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> "ErrorFormatterABC":
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            _cls, value, *_ = args
            e.args = (cls._format(value),)
            raise e

    if not issubclass(cls, ErrorFormatterABC):
        raise TypeError(f"Class `{cls}` must be subtype of `ErrorFormatterABC`.")
    elif not len(cls.__members__):  # type: ignore[attr-defined]
        # empty enum, for class hierarchy
        return func

    return wrapper


class ABCEnumMeta(EnumMeta, ABCMeta):  # noqa: B024
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if getattr(cls, "__error_format__", None) is None:
            raise TypeError(f"Can't instantiate class `{cls.__name__}` without `__error_format__` class attribute.")
        return super().__call__(*args, **kwargs)

    def __new__(cls, clsname: str, superclasses: Tuple[type], attributedict: Dict[str, Any]) -> "ABCEnumMeta":
        res = super().__new__(cls, clsname, superclasses, attributedict)  # type: ignore[arg-type]
        res.__new__ = _pretty_raise_enum(res, res.__new__)  # type: ignore[assignment,arg-type]
        return res


class ErrorFormatterABC(ABC):  # noqa: B024
    """Mixin class that formats invalid value when constructing an enum."""

    __error_format__ = "Invalid option `{0}` for `{1}`. Valid options are: `{2}`."

    @classmethod
    def _format(cls, value: Enum) -> str:
        return cls.__error_format__.format(
            value, cls.__name__, [m.value for m in cls.__members__.values()]  # type: ignore[attr-defined]
        )


class ModeEnum(str, ErrorFormatterABC, PrettyEnum, metaclass=ABCEnumMeta):
    """Enum which prints available values when invalid value has been passed."""

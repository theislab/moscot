from typing import Any

from docrep import DocstringProcessor
from textwrap import dedent

_adata = """\
adata : :class:`anndata.AnnData`
    Annotated data object."""


def inject_docs(**kwargs: Any):  # noqa
    def decorator(obj):
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj):
        obj.__doc__ = dedent(kwargs["__doc__"])
        return obj

    if isinstance(kwargs.get("__doc__", None), str) and len(kwargs) == 1:
        return decorator2

    return decorator


d = DocstringProcessor(
    adata=_adata,
)
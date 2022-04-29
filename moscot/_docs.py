from typing import Any
from textwrap import dedent

from docrep import DocstringProcessor

_adata = """\
adata
    Annotated data object."""
_adatas = """\
adata
    Annotated data objects."""
_adata_x = """"\
    Instance of :class:`anndata.AnnData` containing the data of the source distribution
    """
_adata_y = """"\
    Instance of :class:`anndata.AnnData` containing the data of the target distribution
    """
_solver = """"\
    Instance from :mod:`moscot.solvers` used for solving the Optimal Transport problem
    """
_source = """"\
    Value in :attr:`anndata.AnnData.obs` defining the assignment to the source distribution
    """
_target = """"\
    Value in :attr:`anndata.AnnData.obs` defining the assignment to the target distribution
    """



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
    adatas=_adatas,
    adata_x = _adata_x,
    adata_y = _adata_y,
    solver=_solver,
    source=_source,
    target=_target
)

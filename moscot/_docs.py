from typing import Any
from textwrap import dedent

from docrep import DocstringProcessor

_adata = """\
adata
    Annotated data object."""
_adatas = """\
adatas
    Annotated data objects."""
_adata_x = """\
adata_x
    Instance of :class:`anndata.AnnData` containing the data of the source distribution."""
_adata_y = """\
adata_y
    Instance of :class:`anndata.AnnData` containing the data of the target distribution."""
_solver = """\
solver
    Instance from :mod:`moscot.solvers` used for solving the Optimal Transport problem."""
_source = """\
source
    Value in :attr:`anndata.AnnData.obs` defining the assignment to the source distribution."""
_target = """\
target
    Value in :attr:`anndata.AnnData.obs` defining the assignment to the target distribution."""
_reference = """\
reference
    `reference` in :class:`moscot.problems._subset_policy.StarPolicy`."""
_axis = """\
axis
    Axis along which to group the data."""
_callback = """\
callback
    Custom callback applied to each distribution as preprocessing step. Examples are given in TODO Link Notebook."""
_callback_kwargs = """\
callback_kwargs
    Keyword arguments for `callback`."""
_epsilon = """\
epsilon
    Enropic regularisation parameter."""
_alpha = """\
alpha
    Interpolation parameter between quadratic term and linear term."""
_tau_a = """\
tau_a
    Unbalancedness parameter for left marginal between 0 and 1. `tau_a` equalling 1 means no unbalancedness
    in the source distribution. The limit of `tau_a` going to 0 ignores the left marginals."""
_tau_b = """\
tau_a
    unbalancedness parameter for right marginal between 0 and 1. `tau_b` equalling 1 means no unbalancedness
    in the target distribution. The limit of `tau_b` going to 0 ignores the right marginals."""
_scale_by_marginals = """\
scale_by_marginals
    If `True` the transport map is scaled to be a stochastic matrix by multiplying the resulting mass
            by the inverse of the marginals, TODO maybe EXAMPLE"""
_normalize = """\
normalize
    Whether to normalize the result to 1 after the transport map has been applied."""
_data = """\
data
    - If `data` is a :class:`str` this should correspond to a column in :attr:`anndata.AnnData.obs`.
      The transport map is applied to the subset corresponding to the source distribution
      (if `forward` is `True`) or target distribution (if `forward` is `False`) of that column.
    - If `data` is a :class:npt.ArrayLike the transport map is applied to `data`.
    - If `data` is a :class:`dict` then the keys should correspond to the tuple defining a single optimal
      transport map and the value should be one of the two cases described above."""

_subset = """\
subset
    If `data` is a column in :attr:`anndata.AnnData.obs` the distribution the transport map is applied
    to only puts (uniform) mass on those cells which are in `subset` when filtering for
    :attr:`anndata.AnnData.obs`."""
_marginal_kwargs = """\
marginal_kwargs
    keyword arguments for :meth:`moscot.problems.BirthDeathBaseProblem._estimate_marginals`, i.e. for modeling
    the birth-death process. The keyword arguments
    are either used for :func:`moscot.problems.time._utils.beta`, i.e. one of

        - beta_max: float
        - beta_min: float
        - beta_center: float
        - beta_width: float

    or for :func:`moscot.problems.time._utils.beta`, i.e. one of

        - delta_max: float
        - delta_min: float
        - delta_center: float
        - delta_width: float
"""
_shape = """\
Number of cells in source and target distribution."""
_transport_matrix = """\
Computed transport matrix."""
_converged = """\
Whether the algorihtm converged."""


def inject_docs(**kwargs: Any):
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
    adata_x=_adata_x,
    adata_y=_adata_y,
    solver=_solver,
    source=_source,
    target=_target,
    reference=_reference,
    axis=_axis,
    callback=_callback,
    callback_kwargs=_callback_kwargs,
    epsilon=_epsilon,
    alpha=_alpha,
    tau_a=_tau_a,
    tau_b=_tau_b,
    scale_by_marginals=_scale_by_marginals,
    normalize=_normalize,
    data=_data,
    subset=_subset,
    marginal_kwargs=_marginal_kwargs,
    shape=_shape,
    transport_matrix=_transport_matrix,
    converged=_converged,
)

from typing import Any, TypeVar, Callable, TYPE_CHECKING
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
_scale_cost = """\
scale_cost
    Method to scale cost matrices. If `None` no scaling is applied."""
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
            by the inverse of the marginals, TODO maybe EXAMPLE
"""
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
      transport map and the value should be one of the two cases described above.
"""

_subset = """\
subset
    If `data` is a column in :attr:`anndata.AnnData.obs` the distribution the transport map is applied
    to only puts (uniform) mass on those cells which are in `subset` when filtering for
    :attr:`anndata.AnnData.obs`."""
_marginal_kwargs = """\
marginal_kwargs
    keyword arguments for :meth:`moscot.problems.BirthDeathProblem._estimate_marginals`, i.e. for modeling
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
_a = """\
a
    Specifies the left marginals. If of type :class:`str` the left marginals are taken from
    :attr:`anndata.AnnData.obs` ``[`{a}`]``. If `a` is `None` uniform marginals are used."""
_b = """\
b
    Specifies the right marginals. If of type :class:`str` the right marginals are taken from
    :attr:`anndata.AnnData.obs` ``[`{a}`]``. If `b` is `None` uniform marginals are used."""
_time_key = """\
time_key
    Key in :attr:`anndata.AnnData.obs` which defines the time point each cell belongs to. It is supposed to be
    of numerical data type."""
_spatial_key = """\
spatial_key
    Key in :attr:`anndata.AnnData.obsm` where spatial coordinates are stored."""
_batch_key = """\
batch_key
    If present, specify the batch key of `:class:`anndata.AnnData`."""
_policy = """\
policy
    Defines the rule according to which pairs of distributions are selected to compute the transport map between."""
_key = """\
key
    Key in :attr:`anndata.AnnData.obs` allocating the cell to a certain cell distribution."""
_joint_attr = """\
joint_attr
    Parameter defining how to allocate the data needed to compute the transport maps. If None, the data is read
    from :attr:`anndata.AnnData.X` and for each time point the corresponding PCA space is computed. If
    `joint_attr` is a string the data is assumed to be found in :attr:`anndata.AnnData.obsm`.
    If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of
    :attr:`anndata.AnnData` as a key and the corresponding attribute as a value."""
_split_mass = """\
split_mass
    If `True` the operation is applied to each cell individually."""
_inplace = """\
inplace
    Whether to modify :class:`anndata.AnnData` in place or return the result."""
_online = """\
online
    If `True` the transport matrix is not materialised if it was solved in low-rank mode or with `batch_size` not `None`.
    This reduces memory complexity but increases run time."""
_aggregation_mode = """\
aggregation_mode
    If `aggregation_mode` is `group` the transition probabilities from the groups defined by `source_annotation` are
    returned. If `aggregation_mode` is `cell` the transition probablities for each cell are returned."""
_forward_cell_transition = """\
forward
    If `True` computes transition from `source_annotations` to `target_annotations`, otherwise backward."""
_rank = """\
rank
    Rank of solver. If `-1` standard / full-rank optimal transport is applied."""
_stage = """\
stage
    Stages of subproblems which are to be solved."""
_normalize_cell_transition = """\
normalize
    If `True` the transition matrix is normalized such that it is stochastic. If `forward` is `True`, the transition
    matrix is row-stochastic, otherwise column-stochastic."""
_solve_kwargs = """\
kwargs
    keyword arguments for the backend-specific solver, see NOTEBOOK."""
_heatmap_kwargs = """\
kwargs
    keyword arguments for visualising the transition matrix. Only used if `plot` is `True`."""
_heatmap_plot = """\
plot
    TODO: plots the transition matrix."""
_ott_jax_batch_size = """\
batch_size
    number of data points the matrix-vector products are applied to at the same time. The larger, the more memory
    is required."""
_plotting = """\
figsize
    Size of the figure in inches.
dpi
    Dots per inch.
save
    Whether to save the plot."""
_input_plotting = """\
input
    The :class:`anndata.AnnData` instance(s) where the results of the corresponding method of the moscot problem
    instance is saved. Alternatively, the instance of the moscot problem can be passed, too.
    """
_key_stored = """\
key_stored
    A key of :class:`anndata.AnnData` where the results of the corresponding method of the moscot problem instance
    is saved."""
_cont_cmap = """\
cont_cmap
    Colormap for continuous annotations, see :class:`matplotlib.colors.Colormap`."""
_cbar_kwargs = """\
cbar_kwargs
    Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`."""
_ax = """\
ax
    Axes, :class:`matplotlib.axes.Axes`."""
_key_added_plotting = """\
key_added
    Key where the results for the corresponding plotting functions are stored."""
_constant_fill_value = """\
constant_fill_value
    Value for cells which do not belong to the distributions which are to be colorized."""
_plot_time_points = """\
time_points
    Time points which are colorised in the embedding plot."""
# returns
_alignment_mixin_returns = """\
If ``inplace = False``, returns a :class:`numpy.ndarray` with aligned coordinates.

Otherwise, modifies the ``adata`` object with the following key:

    - :attr:`anndata.AnnData.obsm` ``['{key_added}']`` - the above mentioned :class:`numpy.ndarray`.
"""

RT = TypeVar("RT")  # return type
O = TypeVar("O")  # noqa: E741, object type


def inject_docs(**kwargs: Any) -> Callable[[Callable[..., RT]], Callable[..., RT]]:  # noqa: D103
    def decorator(obj: O) -> O:  # noqa: E741
        if TYPE_CHECKING:
            assert isinstance(obj.__doc__, str)
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj: O) -> O:  # noqa: E741
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
    scale_cost=_scale_cost,
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
    a=_a,
    b=_b,
    time_key=_time_key,
    spatial_key=_spatial_key,
    batch_key=_batch_key,
    policy=_policy,
    key=_key,
    joint_attr=_joint_attr,
    split_mass=_split_mass,
    inplace=_inplace,
    alignment_mixin_returns=_alignment_mixin_returns,
    online=_online,
    aggregation_mode=_aggregation_mode,
    forward_cell_transition=_forward_cell_transition,
    rank=_rank,
    stage=_stage,
    normalize_cell_transition=_normalize_cell_transition,
    solve_kwargs=_solve_kwargs,
    heatmap_kwargs=_heatmap_kwargs,
    heatmap_plot=_heatmap_plot,
    ott_jax_batch_size=_ott_jax_batch_size,
    plotting=_plotting,
    input_plotting=_input_plotting,
    cont_cmap=_cont_cmap,
    cbar_kwargs=_cbar_kwargs,
    key_stored=_key_stored,
    ax=_ax,
    key_added_plotting=_key_added_plotting,
    constant_fill_value=_constant_fill_value,
)

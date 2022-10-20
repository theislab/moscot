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
_callback = """\
callback
    Custom callback applied to each distribution as pre-processing step. Examples are given in TODO Link Notebook."""
_callback_kwargs = """\
callback_kwargs
    Keyword arguments for `callback`."""
_epsilon = """\
epsilon
    Entropic regularisation parameter."""
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
tau_b
    unbalancedness parameter for right marginal between 0 and 1. `tau_b` equalling 1 means no unbalancedness
    in the target distribution. The limit of `tau_b` going to 0 ignores the right marginals."""
_scale_by_marginals = """\
scale_by_marginals
    If `True` the transport map is scaled to be a stochastic matrix by multiplying the resulting mass
    by the inverse of the marginals, TODO maybe EXAMPLE."""
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
    Subset of :attr:`anndata.AnnData.obs` ``['{key}']`` values of which the policy is to be applied to."""
_marginal_kwargs = """\
marginal_kwargs
    keyword arguments for :meth:`moscot.problems.BirthDeathProblem._estimate_marginals`, i.e.
    for modeling the birth-death process. The keyword arguments
    are either used for :func:`moscot.problems.time._utils.beta`, i.e. one of:

        - beta_max: float
        - beta_min: float
        - beta_center: float
        - beta_width: float

    or for :func:`moscot.problems.time._utils.beta`, i.e. one of:

        - delta_max: float
        - delta_min: float
        - delta_center: float
        - delta_width: float
"""
_shape = """\
shape
    Number of cells in source and target distribution."""
_transport_matrix = """\
transport_matrix
    Computed transport matrix."""
_converged = """\
converged
    Whether the algorithm converged."""
_a = """\
a
    Specifies the left marginals. If of type :class:`str` the left marginals are taken from
    :attr:`anndata.AnnData.obs` ``['{a}']``. If `a` is `None` uniform marginals are used."""
_b = """\
b
    Specifies the right marginals. If of type :class:`str` the right marginals are taken from
    :attr:`anndata.AnnData.obs` ``['{b}']``. If `b` is `None` uniform marginals are used."""
_time_key = """\
time_key
    Time point key in :attr:`anndata.AnnData.obs`."""
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
    Key in :attr:`anndata.AnnData.obs` allocating the cell to a certain cell distribution (e.g. batch)."""
_joint_attr = """\
joint_attr

    - If `None`, PCA on :attr:`anndata.AnnData.X` is computed.
    - If `str`, it must refer to a key in :attr:`anndata.AnnData.obsm`.
    - If `dict`, the dictionary stores `attr` (attribute of :class:`anndata.AnnData`) and `key`
      (key of :class:`anndata.AnnData` ``['{attr}']``).
"""
_GW_x = """\
GW_x

    - If empty , cost matrix must be provided in :attr:`anndata.AnnData.obsp`.
    - If `str`, it must refer to a key in :attr:`anndata.AnnData.obsm`.
    - If `dict`, the dictionary stores `attr` (attribute of :class:`anndata.AnnData`) and `key`
      (key of :class:`anndata.AnnData` ``['{attr}']``).
"""
_GW_y = """\
GW_y

    - If empty, cost matrix must be provided in :attr:`anndata.AnnData.obsp`.
    - If `str`, it must refer to a key in :attr:`anndata.AnnData.obsm`.
    - If `dict`, the dictionary stores `attr` (attribute of :class:`anndata.AnnData`) and `key`
      (key of :class:`anndata.AnnData` ``['{attr}']``).
"""
_split_mass = """\
split_mass
    If `True` the operation is applied to each cell individually."""
_inplace = """\
inplace
    Whether to modify :class:`anndata.AnnData` in place or return the result."""
_online = """\
online
    If `True` the transport matrix is not materialised if it was solved in low-rank mode or with `batch_size != None`.
    This reduces memory complexity but increases run time."""
_rank = """\
rank
    Rank of solver. If `-1` standard / full-rank optimal transport is applied."""
_stage = """\
stage
    Stages of subproblems which are to be solved."""
_solve_kwargs = """\
kwargs
    keyword arguments for the backend-specific solver, TODO see NOTEBOOK."""
_ott_jax_batch_size = """\
batch_size
    number of data points the matrix-vector products are applied to at the same time. The larger, the more memory
    is required."""
_alignment_mixin_returns = """\
If ``inplace = False``, returns a :class:`numpy.ndarray` with aligned coordinates.

Otherwise, modifies the :class:`anndata.AnnData` instance with the following key:

    - :attr:`anndata.AnnData.obsm` ``['{key_added}']`` - the above mentioned :class:`numpy.ndarray`.
"""
_initializer_lin = """\
initializer
    Initializer to use for the problem.
    If not low rank, available options are

        - `default` (constant scalings)
        - `gaussian` :cite:`thornton:22`
        - `sorting` :cite:`thornton:22`

    If low rank, available options are:

        - `random`
        - `rank2` :cite:`scetbon:21a`
        - `k-means` :cite:`scetbon:22b`
        - `generalized-k-means` :cite:`scetbon:22b`

    If `None`, the default for not low rank is `default`, for low rank it is `random`.
"""
_initializer_quad = """\
initializer
    Initializer to use for the problem.
    If not low rank, the standard initializer is used (outer product of marginals).
    If low rank, available options are:

        - `random`
        - `rank2` :cite:`scetbon:21a`
        - `k-means` :cite:`scetbon:22b`
        - `generalized-k-means` :cite:`scetbon:22b`:

    If `None`, the low-rank initializer will be selected based on how the data is passed.
    If the cost matrix is passed (instead of the data), the random initializer is used,
    otherwise the K-means initializer.
"""
_initializer_kwargs = """\
initializer_kwargs
    keyword arguments for the initializer.
"""
_jit = """\
jit
    if True, automatically jits (just-in-time compiles) the function upon first call.
"""
_sinkhorn_kwargs = """\
threshold
    Tolerance used to stop the Sinkhorn iterations. This is
    typically the deviation between a target marginal and the marginal of the
    current primal solution when either or both tau_a and tau_b are 1.0
    (balanced or semi-balanced problem), or the relative change between two
    successive solutions in the unbalanced case.
lse_mode
    ``True`` for log-sum-exp computations, ``False`` for kernel
      multiplication.
norm_error
    Power used to define p-norm of error for stopping criterion, see ``threshold``.
inner_iterations
    The Sinkhorn error is not recomputed at each iteration but every ``inner_iterations`` instead.
min_iterations
    The minimum number of Sinkhorn iterations carried out before the error is computed and monitored.
max_iterations
    The maximum number of Sinkhorn iterations.
"""
_sinkhorn_lr_kwargs = """\
gamma
    Only in low-rank setting: the (inverse of the) gradient step size used by the mirror descent algorithm
    (:cite:`scetbon:22b`).
gamma_rescale
    Only in low-rank setting: whether to rescale :math:`\\gamma` every iteration as described in :cite:`scetbon:22b`.
"""
_gw_kwargs = """\
min_iterations
    Minimal number of iterations.
max_iterations
    Maximal number of iterations.
threshold
    Threshold used as convergence criterion.
warm_start
    Whether to initialize (low-rank) Sinkhorn calls using values
    from the previous iteration. If `None`, warm starts are not used for
    standard Sinkhorn, but used for low-rank Sinkhorn.
"""


d = DocstringProcessor(
    adata=_adata,
    adatas=_adatas,
    adata_x=_adata_x,
    adata_y=_adata_y,
    solver=_solver,
    source=_source,
    target=_target,
    reference=_reference,
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
    GW_x=_GW_x,
    GW_y=_GW_y,
    split_mass=_split_mass,
    inplace=_inplace,
    alignment_mixin_returns=_alignment_mixin_returns,
    online=_online,
    rank=_rank,
    stage=_stage,
    solve_kwargs=_solve_kwargs,
    ott_jax_batch_size=_ott_jax_batch_size,
    initializer_lin=_initializer_lin,
    initializer_quad=_initializer_quad,
    initializer_kwargs=_initializer_kwargs,
    jit=_jit,
    sinkhorn_kwargs=_sinkhorn_kwargs,
    sinkhorn_lr_kwargs=_sinkhorn_lr_kwargs,
    gw_kwargs=_gw_kwargs,
)

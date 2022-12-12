from docrep import DocstringProcessor

_cell_trans_params = """\
source
    Key identifying the source distribution.
target
    Key identifying the target distribution.
source_groups
    Can be one of the following:

        - if `source_groups` is of type :class:`str` this should correspond to a key in
          :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
          unique values in :attr:`anndata.AnnData.obs` ``['{source_groups}']``.
        - if `target_groups` is of type :class:`dict`, its key should correspond to a key in
          :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories present in
          :attr:`anndata.AnnData.obs` ``['{source_groups.keys()[0]}']``. The order of the list determines the order
          in the transition matrix.

target_groups
    Can be one of the following:

        - if `target_groups` is of type :class:`str` this should correspond to a key in
          :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
          unique values in :attr:`anndata.AnnData.obs` ``['{target_groups}']``.
        - if `target_groups` is of :class:`dict`, its key should correspond to a key in
          :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories present in
          :attr:`anndata.AnnData.obs` ``['{target_groups.keys()[0]}']``. The order of the list determines the order
          in the transition matrix.
"""
_key = """\
key
    Key in :attr:`anndata.AnnData.obs` allocating the cell to a certain cell distribution (e.g. batch)."""
_aggregation_mode = """\
aggregation_mode

    - `group`: transition probabilities from the groups defined by `source_annotation` are returned.
    - `cell`: the transition probabilities for each cell are returned.
"""
_other_key = """\
other_key
    Key in :attr:`anndata.AnnData.obs` allocating the cell to a certain cell distribution (e.g. batch).
"""
_other_adata = """\
adata
    Annotated data object.
"""
_ott_jax_batch_size = """\
batch_size
    number of data points the matrix-vector products are applied to at the same time. The larger, the more memory
    is required.
"""
_key_added_plotting = """\
key_added
    Key in :attr:`anndata.AnnData.uns` and/or :attr:`anndata.AnnData.obs` where the results
    for the corresponding plotting functions are stored.
    See TODO Notebook for how :mod:`moscot.plotting` works.
"""
_return_cell_transition = """\
retun_cell_transition
    Transition matrix of cells or groups of cells.
"""
_notes_cell_transition = """\
To visualise the results, see :func:`moscot.pl.cell_transition`.
"""
_normalize = """\
normalize
    If `True` the transition matrix is normalized such that it is stochastic. If `forward` is `True`, the transition
    matrix is row-stochastic, otherwise column-stochastic.
"""
_forward_cell_transition = """\
forward
    If `True` computes transition from `source_annotations` to `target_annotations`, otherwise backward.
"""
_return_data = """\
return_data
    Whether to return the data.
"""
_return_all = """\
return_all
    If `True` returns all the intermediate masses if pushed through multiple transport plans, returned as a
    dictionary.
"""
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
    Subset of :attr:`anndata.AnnData.obs` ``['{key}']`` values of which the policy is to be applied to.
"""
_source = """\
source
    Value in :attr:`anndata.AnnData.obs` defining the assignment to the source distribution.
"""
_target = """\
target
    Value in :attr:`anndata.AnnData.obs` defining the assignment to the target distribution.
"""
_scale_by_marginals = """\
scale_by_marginals
    If `True` the transport map is scaled to be a stochastic matrix by multiplying the resulting mass
    by the inverse of the marginals, TODO maybe EXAMPLE.
"""
_return_push_pull = """
Depending on `key_added` updates `adata` or returns the result:

    - In the former case all intermediate results are saved in :attr:`anndata.AnnData.obs`.
    - In the latter case all intermediate step results are returned if `return_all` is `True`,
      otherwise only the distribution at `source` is returned.
"""
_restrict_to_existing = """\
restrict_to_existing
    TODO.
"""
_order_annotations = """\
order_annotations
    Order of the annotations in the final plot, from top to bottom.
"""
_threshold = """\
threshold
    If not `None`, set all entries below `threshold` to 0.
"""
_backend = """\
backend
    Which backend to use for solving Optimal Transport problems.
"""
_kwargs_divergence = """\
kwargs
    Keyword arguments to solve the underlying Optimal Transport problem, see example TODO.
"""
_start = """\
start
    Time point corresponding to the early distribution.
"""
_end = """\
end
    Time point corresponding to the late distribution.
"""
_intermediate = """\
intermediate
    Time point corresponding to the intermediate distribution.
"""
_intermediate_interpolation = """\
intermediate
    Time point corresponding to the intermediate distribution which is to be interpolated.
"""
_seed_sampling = """\
seed
    Random seed for sampling from the transport matrix.
"""
_interpolation_parameter = """\
interpolation_parameter
    Interpolation parameter determining the weight of the source and the target distribution. If `None`
    it is linearly inferred from `source`, `intermediate`, and `target`.
"""
_account_for_unbalancedness = """\
account_for_unbalancedness
    If `True` unbalancedness is accounted for by assuming exponential growth and death of cells.
"""
_n_interpolated_cells = """\
n_interpolated_cells
    Number of generated interpolated cell. If `None` the number of data points in the `intermediate`
    distribution is taken.
"""
_seed_interpolation = """\
seed
    Random seed for generating randomly interpolated cells.
"""
_time_batch_distance = """\
time
    Time point corresponding to the cell distribution which to compute the batch distances within.
"""
_batch_key_batch_distance = """\
batch_key
    Key in :attr:`anndata.AnnData.obs` storing which batch each cell belongs to.
"""
_use_posterior_marginals = """\
posterior_marginals
    Whether to use posterior marginals (posterior growth rates). This requires the problem to be solved.
    If `False`, prior marginals are used.
"""

d_mixins = DocstringProcessor(
    cell_trans_params=_cell_trans_params,
    key=_key,
    forward_cell_transition=_forward_cell_transition,
    aggregation_mode=_aggregation_mode,
    other_key=_other_key,
    other_adata=_other_adata,
    ott_jax_batch_size=_ott_jax_batch_size,
    key_added_plotting=_key_added_plotting,
    return_cell_transition=_return_cell_transition,
    notes_cell_transition=_notes_cell_transition,
    normalize=_normalize,
    # TODO(MUCDK): duplicate
    forward=_forward_cell_transition,
    return_data=_return_data,
    return_all=_return_all,
    return_push_pull=_return_push_pull,
    data=_data,
    subset=_subset,
    source=_source,
    target=_target,
    scale_by_marginals=_scale_by_marginals,
    restrict_to_existing=_restrict_to_existing,
    order_annotations=_order_annotations,
    threshold=_threshold,
    backend=_backend,
    kwargs_divergence=_kwargs_divergence,
    start=_start,
    end=_end,
    intermediate=_intermediate,
    intermediate_interpolation=_intermediate_interpolation,
    seed_sampling=_seed_sampling,
    interpolation_parameter=_interpolation_parameter,
    n_interpolated_cells=_n_interpolated_cells,
    seed_interpolatiob=_seed_interpolation,
    time_batch_distance=_time_batch_distance,
    batch_key_batch_distance=_batch_key_batch_distance,
    use_posterior_marginals=_use_posterior_marginals,
)

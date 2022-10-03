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
        :attr:`anndata.AnnData.obs` and its value to a subset of categories present in
        :attr:`anndata.AnnData.obs` ``['{source_groups.keys()[0]}']``.
target_groups
    Can be one of the following
        - if `target_groups` is of type :class:`str` this should correspond to a key in
        :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
        unique values in :attr:`anndata.AnnData.obs` ``['{target_groups}']``.
        - if `target_groups` is of :class:`dict`, its key should correspond to a key in
        :attr:`anndata.AnnData.obs` and its value to a subset of categories present in
        :attr:`anndata.AnnData.obs` ``['{target_groups.keys()[0]}']``.
"""
_key = """\
key
    Key in :attr:`anndata.AnnData.obs` allocating the cell to a certain cell distribution (e.g. batch)."""
_forward_cell_transition = """\
forward
    If `True` computes transition from `source_annotations` to `target_annotations`, otherwise backward."""
_aggregation_mode = """\
aggregation_mode
    - `group`: transition probabilities from the groups defined by `source_annotation` are returned.
    - `cell`: the transition probabilities for each cell are returned."""
_online = """\
online
    If `True` the transport matrix is not materialised if it was solved in low-rank mode or with `batch_size != None`.
    This reduces memory complexity but increases run time."""
_other_key = """\
other_key
    Key in :attr:`anndata.AnnData.obs` allocating the cell to a certain cell distribution (e.g. batch)."""
_other_adata = """\
adata
    Annotated data object."""
_ott_jax_batch_size = """\
batch_size
    number of data points the matrix-vector products are applied to at the same time. The larger, the more memory
    is required."""
_key_added_plotting = """\
key_added
    Key in :attr:`anndata.AnnData.uns` and/or :attr:`anndata.AnnData.obs` where the results
    for the corresponding plotting functions are stored.
    See TODO Notebook for how :mod:`moscot.plotting` works."""
_return_cell_transition = """\
Transition matrix of cells or groups of cells."""
_notes_cell_transition = """\
To visualise the results, see :func:`moscot.pl.cell_transition`.
"""
_normalize = """\
normalize
    If `True` the transition matrix is normalized such that it is stochastic. If `forward` is `True`, the transition
    matrix is row-stochastic, otherwise column-stochastic."""
_forward_cell_transition = """\
forward
    If `True` computes transition from `source_annotations` to `target_annotations`, otherwise backward."""
_return_data = """\
return_data
    Whether to return the data."""
_return_all = """\
return_all
    If `True` returns all the intermediate masses if pushed through multiple transport plans, returned as a
    dictionary."""
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
    Subset of :attr:`anndata.AnnData.obs` ``['{key}']`` values of which the policy is to be applied to."""
_source = """\
source
    Value in :attr:`anndata.AnnData.obs` defining the assignment to the source distribution."""
_target = """\
target
    Value in :attr:`anndata.AnnData.obs` defining the assignment to the target distribution."""
_scale_by_marginals = """\
scale_by_marginals
    If `True` the transport map is scaled to be a stochastic matrix by multiplying the resulting mass
    by the inverse of the marginals, TODO maybe EXAMPLE."""
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


d_mixins = DocstringProcessor(
    cell_trans_params=_cell_trans_params,
    key=_key,
    forward_cell_transition=_forward_cell_transition,
    aggregation_mode=_aggregation_mode,
    online=_online,
    other_key=_other_key,
    other_adata=_other_adata,
    ott_jax_batch_size=_ott_jax_batch_size,
    key_added_plotting=_key_added_plotting,
    return_cell_transition=_return_cell_transition,
    notes_cell_transition=_notes_cell_transition,
    normalize=_normalize,
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
)

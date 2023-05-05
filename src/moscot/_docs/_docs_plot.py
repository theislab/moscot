from docrep import DocstringProcessor

###############################################################################
# plotting.cell_transition
# input
_desc_cell_transition = """\
In order to run this function the corresponding method `cell_transition`
of the :mod:`moscot.problems` instance must have been run, see `Notes`.
"""
_transition_labels_cell_transition = """\
row_labels
    If present, used labels to annotate the rows of the transition matrix.
col_labels
    If present, used labels to annotate the columns of the transition matrix.
annotate
    If `None`, the heatmap does not show the values of the transition matrix. Otherwise,
    expects formatting value.
"""
_cbar_kwargs_cell_transition = """\
cbar_kwargs
    Keyword arguments for :meth:`~matplotlib.figure.Figure.colorbar`."""
# return cell transition
_return_cell_transition = """\
Heatmap of cell transition matrix.
"""
# notes cell transition
_notes_cell_transition = """\
This function looks for the following data in the :class:`~anndata.AnnData` object
which is passed or saved as an attribute of :class:`moscot.base.problems.CompoundProblem`.

    - `transition_matrix`
    - `source`
    - `target`
    - `source_groups`
    - `target_groups`
"""
_fontsize = """\
fontsize
    Fontsize of annotation."""

###############################################################################
# plotting.sankey
# input
_desc_sankey = """\
In order to run this function the corresponding method `sankey`
of the :mod:`moscot.problems` instance must have been run, see `Notes`.
"""
_captions_sankey = """\
captions
    TODO."""
_colors_dict_sankey = """\
colors_dict
    TODO."""
# return sankey
_return_sankey = """\
:class:`matplotlib.figure.Figure` Sankey diagram.
"""
# notes sankey
_notes_sankey = """\
This function looks for the following data in the :class:`~anndata.AnnData` object
which is passed or saved as an attribute of :class:`moscot.base.problems.CompoundProblem`.

    - `transition_matrices`
    - `captions`
    - `key`
"""
_alpha_transparency = """\
alpha
    Transparency value.
"""
_interpolate_color = """\
interpolate_color
    Whether the color is continuously interpolated.
"""
_sankey_kwargs = """\
kwargs
    Keyword arguments for :func:`matplotlib.pyplot.fill_between`.
"""
###############################################################################
# plotting.push/pull
# input
_desc_push_pull = """\
TODO.
"""
_result_key_push_pull = """\
result_key
    Key in :attr:`anndata.AnnData.uns` and/or :attr:`anndata.AnnData.obs` where the results of the
    plotting functions are stored."""
_fill_value_push_pull = """\
fill_value_push_pull
    Fill value for observations not present in selected batches."""
_time_points_push_pull = """\
time_points
    Time points colored in the embedding plot."""
_basis_push_pull = """\
basis
    Basis of the embedding, saved in :attr:`anndata.AnnData.obsm`.
"""
_scale_push_pull = """\
scale
    Whether to linearly scale the distribution.
"""
# return push/pull
_return_push_pull = """\
Scatterplot in `basis` coordinates.
"""
# notes push/pull
_notes_push_pull = """\
This function looks for the following data in the :class:`~anndata.AnnData` object
which is passed or saved as an attribute of :class:`moscot.base.problems.CompoundProblem`.

    - `temporal_key`
"""

###############################################################################
# general input
_input_plotting = """\
inp
    An instance of :class:`~anndata.AnnData` where the results of the corresponding method
    of the :mod:`moscot.problems` instance is saved.
    Alternatively, the instance of the moscot problem can be passed, too."""
_uns_key = """\
uns_key
    Key of :attr:`anndata.AnnData.uns` where the results of the corresponding method
    of the moscot problem instance is saved."""
_cmap = """\
cmap
    Colormap for continuous annotations, see :class:`~matplotlib.colors.Colormap`."""
_title = """\
title
    Title of the plot.
"""
_dot_scale_factor = """\
dot_scale_factor
    If `time_points` is not `None`, `dot_scale_factor` increases the size of the dots by this factor.
"""
_na_color = """\
na_color
    Color to use for null or masked values. Can be anything matplotlib accepts as a color.
"""
_suptitle_fontsize = """
suptitle_fontsize
    Fontsize of the suptitle.
"""
###############################################################################
# general output
_return_fig = """\
return_fig
    Whether to return the figure."""
_ax = f"""\
ax
    Axes.
{_return_fig}"""
_figsize_dpi_save = f"""\
figsize
    Size of the figure in inches.
dpi
    Dots per inch.
save
    Path where to save the plot. If `None`, the plot is not saved.
{_ax}"""

d_plotting = DocstringProcessor(
    desc_cell_transition=_desc_cell_transition,
    transition_labels_cell_transition=_transition_labels_cell_transition,
    cbar_kwargs_cell_transition=_cbar_kwargs_cell_transition,
    return_cell_transition=_return_cell_transition,
    notes_cell_transition=_notes_cell_transition,
    desc_sankey=_desc_sankey,
    captions_sankey=_captions_sankey,
    colors_dict_sankey=_colors_dict_sankey,
    return_sankey=_return_sankey,
    notes_sankey=_notes_sankey,
    desc_push_pull=_desc_push_pull,
    result_key_push_pull=_result_key_push_pull,
    fill_value_push_pull=_fill_value_push_pull,
    time_points_push_pull=_time_points_push_pull,
    basis_push_pull=_basis_push_pull,
    return_push_pull=_return_push_pull,
    notes_push_pull=_notes_push_pull,
    input_plotting=_input_plotting,
    uns_key=_uns_key,
    cmap=_cmap,
    title=_title,
    return_fig=_return_fig,
    ax=_ax,
    figsize_dpi_save=_figsize_dpi_save,
    fontsize=_fontsize,
    alpha_transparency=_alpha_transparency,
    interpolate_color=_interpolate_color,
    sankey_kwargs=_sankey_kwargs,
    na_color=_na_color,
    dot_scale_factor=_dot_scale_factor,
    scale_push_pull=_scale_push_pull,
    suptitle_fontsize=_suptitle_fontsize,
)

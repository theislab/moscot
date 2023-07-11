import pathlib
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

import matplotlib as mpl

from anndata import AnnData

from moscot import _constants
from moscot.plotting._utils import (
    _create_col_colors,
    _heatmap,
    _input_to_adatas,
    _plot_temporal,
    _sankey,
    get_plotting_vars,
)

if TYPE_CHECKING:
    from moscot.base.problems import CompoundProblem
    from moscot.problems import LineageProblem, SpatioTemporalProblem, TemporalProblem

__all__ = ["cell_transition", "sankey", "push", "pull"]


def cell_transition(
    obj: Union[AnnData, Tuple[AnnData, AnnData], "CompoundProblem"],  # type: ignore[type-arg]
    key: str = _constants.CELL_TRANSITION,
    row_label: Optional[str] = None,
    col_label: Optional[str] = None,
    annotate: Optional[str] = "{x:.2f}",
    fontsize: float = 7.0,
    cmap: Union[str, mpl.colors.Colormap] = "viridis",
    cbar_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
    ax: Optional[mpl.axes.Axes] = None,
    return_fig: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """Plot an aggregate cell transition matrix.

    .. seealso::
        - See :doc:`../notebooks/examples/plotting/200_cell_transitions` on how to
          :meth:`compute <moscot.problems.time.TemporalProblem.cell_transition>` and plot the cell transitions.

    Parameters
    ----------
    obj
        Object containing the :meth:`cell transition <moscot.problems.time.TemporalProblem.cell_transition>` data.
        Valid options are:

        - :class:`~anndata.AnnData` - annotated data object containing the data.
        - :class:`tuple` - source and target :class:`~anndata.AnnData` objects.
        - :class:`~moscot.base.problems.CompoundProblem` - one of the :mod:`moscot.problems`.
    key
        Key in :attr:`uns['moscot_results'] <anndata.AnnData.uns>` where the cell transition data is stored.
    row_label
        Label for the rows of the transition matrix.
    col_label
        Label for the columns of the transition matrix.
    annotate
        Format used when annotating the cells. If :obj:`None`, don't annotate the cells.
    fontsize
        Font size for the annotations.
    cmap
        Colormap of the heatmap.
    cbar_kwargs
        Keyword arguments for :meth:`~matplotlib.figure.Figure.colorbar`.
    ax
        Ax used for plotting. If :obj:`None`, create a new one.
    return_fig
        Whether to return the figure.
    figsize
        Size of the figure.
    dpi
        Dots per inch.
    save
        Path where to save the figure.
    kwargs
        Keyword arguments for :meth:`~matplotlib.axes.Axes.text`.

    Returns
    -------
    If ``return_fig = True``, returns and plots the figure. Otherwise, just plots the figure.
    """
    adata1, adata2 = _input_to_adatas(obj)
    data = get_plotting_vars(adata1, _constants.CELL_TRANSITION, key=key)

    fig = _heatmap(
        row_adata=adata1,
        col_adata=adata2,
        transition_matrix=data["transition_matrix"],
        row_annotation=data["source_groups"]
        if isinstance(data["source_groups"], str)
        else next(iter(data["source_groups"])),
        col_annotation=data["target_groups"]
        if isinstance(data["target_groups"], str)
        else next(iter(data["target_groups"])),
        row_annotation_label=data["source"] if row_label is None else row_label,
        col_annotation_label=data["target"] if col_label is None else col_label,
        cont_cmap=cmap,
        annotate_values=annotate,
        fontsize=fontsize,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        save=save,
        cbar_kwargs=cbar_kwargs,
        **kwargs,
    )
    return fig if return_fig else None


def sankey(
    obj: Union[AnnData, "TemporalProblem", "LineageProblem", "SpatioTemporalProblem"],
    key: str = _constants.SANKEY,
    captions: Optional[List[str]] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, float]] = None,
    alpha: float = 1.0,
    interpolate_color: bool = False,
    cmap: Union[str, mpl.colors.Colormap] = "viridis",
    ax: Optional[mpl.axes.Axes] = None,
    return_fig: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
    **kwargs: Any,
) -> Optional[mpl.figure.Figure]:
    """Plot a `Sankey diagram <https://en.wikipedia.org/wiki/Sankey_diagram>`_ between cells across time points.

    .. seealso::
        - See :doc:`../notebooks/examples/plotting/300_sankey` on how to
          :meth:`compute <moscot.problems.time.TemporalProblem.sankey>` and plot the Sankey diagram.

    Parameters
    ----------
    obj
        Object containing the :meth:`Sankey diagram <moscot.problems.time.TemporalProblem.sankey>` data.
        Valid options are:

        - :class:`~anndata.AnnData` - annotated data object containing the data.
        - :class:`~moscot.problems.time.TemporalProblem`/:class:`~moscot.problems.time.LineageProblem`/
          :class:`~moscot.problems.spatiotemporal.SpatioTemporalProblem` - one of the
          :mod:`temporal problems <moscot.problems>`.
    key
        Key in :attr:`uns['moscot_results'] <anndata.AnnData.uns>` where the cell transition data is stored.
    captions
        TODO(MUCDK)
    title
        Title of the figure.
    colors
        TODO(MUCDK)
    alpha
        Transparency value in :math:`[0, 1]`; :math:`0` (transparent) and :math:`1` (opaque).
    interpolate_color
        Whether the color is continuously interpolated from the source to the target.
    cmap
        Colormap of the diagram.
    ax
        Ax used for plotting. If :obj:`None`, create a new one.
    return_fig
        Whether to return the figure.
    figsize
        Size of the figure.
    dpi
        Dots per inch.
    save
        Path where to save the figure.
    kwargs
        Keyword arguments for :meth:`~matplotlib.axes.Axes.fill_between`.

    Returns
    -------
    If ``return_fig = True``, returns and plots the figure. Otherwise, just plots the figure.
    """
    adata, _ = _input_to_adatas(obj)
    data = get_plotting_vars(adata, _constants.SANKEY, key=key)

    fig = _sankey(
        adata=adata,
        key=data["key"],
        transition_matrices=data["transition_matrices"],
        captions=data["captions"] if captions is None else captions,
        colorDict=colors,
        cont_cmap=cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        alpha=alpha,
        interpolate_color=interpolate_color,
        **kwargs,
    )
    if save:
        fig.figure.savefig(save)
    return fig if return_fig else None


def push(
    obj: Union[AnnData, "CompoundProblem"],  # type: ignore[type-arg]
    key: str = _constants.PUSH,
    basis: str = "umap",
    time_points: Optional[Sequence[float]] = None,
    fill_value: float = np.nan,
    scale: bool = True,
    dot_scale_factor: float = 2.0,
    cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
    na_color: str = "#e8ebe9",
    title: Optional[Union[str, List[str]]] = None,
    suptitle: Optional[str] = None,
    suptitle_fontsize: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    return_fig: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
    **kwargs: Any,
) -> Optional[mpl.figure.Figure]:
    """Plot the push-forward distribution.

    .. seealso::
        - See :doc:`../notebooks/examples/plotting/100_push_pull` on how to
          :meth:`compute <moscot.base.problems.CompoundProblem.push>` and plot the push-forward distribution.

    Parameters
    ----------
    obj
        Object containing the :meth:`push-forward <moscot.base.problems.CompoundProblem.push>` distribution.
        Valid options are:

        - :class:`~anndata.AnnData` - annotated data object.
        - :class:`~moscot.base.problems.CompoundProblem` - one of the :mod:`moscot.problems`.
    key
        Key in :attr:`uns['moscot_results'] <anndata.AnnData.uns>` where the push-forward distribution is stored.
    basis
        Key in :attr:`~anndata.AnnData.obsm` where the embedding is stored.
    time_points
        Time points in :attr:`~anndata.AnnData.obs` to highlight.
    fill_value
        Fill value for observations not present in selected batches
    scale
        Whether to linearly scale the distribution.
    dot_scale_factor
        Scale factor for the ``time_points``.
    cmap
        Colormap for continuous observations.
    na_color
        Color for NaN values.
    title
        Title of the figure.
    suptitle
        Suptitle of the figure.
    suptitle_fontsize
        Font size of the suptitle.
    ax
        Ax used for plotting. If :obj:`None`, create a new one.
    return_fig
        Whether to return the figure.
    figsize
        Size of the figure.
    dpi
        Dots per inch.
    save
        Path where to save the figure.
    kwargs
        Keyword arguments for :func:`~scanpy.pl.embedding`.

    Returns
    -------
    If ``return_fig = True``, returns and plots the figure. Otherwise, just plots the figure.
    """
    adata, _ = _input_to_adatas(obj)
    if key not in adata.obs:
        raise KeyError(f"No data found in `adata.obs[{key!r}]`.")

    data = get_plotting_vars(adata, _constants.PUSH, key=key)
    if data["data"] is not None and data["subset"] is not None and cmap is None:
        cmap = _create_col_colors(adata, data["data"], data["subset"])

    fig = _plot_temporal(
        adata=adata,
        temporal_key=data["temporal_key"],
        key_stored=key,
        source=data["source"],
        target=data["target"],
        categories=data["subset"],
        push=True,
        time_points=time_points,
        basis=basis,
        constant_fill_value=fill_value,
        scale=scale,
        save=save,
        cont_cmap=cmap,
        dot_scale_factor=dot_scale_factor,
        na_color=na_color,
        title=title,
        suptitle=suptitle,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        suptitle_fontsize=suptitle_fontsize,
        **kwargs,
    )
    return fig if return_fig else None


def pull(
    obj: Union[AnnData, "CompoundProblem"],  # type: ignore[type-arg]
    key: str = _constants.PULL,
    basis: str = "umap",
    time_points: Optional[Sequence[float]] = None,
    fill_value: float = np.nan,
    scale: bool = True,
    dot_scale_factor: float = 2.0,
    cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
    na_color: str = "#e8ebe9",
    title: Optional[Union[str, List[str]]] = None,
    suptitle: Optional[str] = None,
    suptitle_fontsize: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    return_fig: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
    **kwargs: Any,
) -> Optional[mpl.figure.Figure]:
    """Plot the pull-back distribution.

    .. seealso::
        - See :doc:`../notebooks/examples/plotting/100_push_pull` on how to
          :meth:`compute <moscot.base.problems.CompoundProblem.pull>` and plot the pull-back distribution.

    Parameters
    ----------
    obj
        Object containing the :meth:`pull-back <moscot.base.problems.CompoundProblem.pull>` distribution.
        Valid options are:

        - :class:`~anndata.AnnData` - annotated data object.
        - :class:`~moscot.base.problems.CompoundProblem` - one of the :mod:`moscot.problems`.
    key
        Key in :attr:`uns['moscot_results'] <anndata.AnnData.uns>` where the pull-back distribution is stored.
    basis
        Key in :attr:`~anndata.AnnData.obsm` where the embedding is stored.
    time_points
        Time points in :attr:`~anndata.AnnData.obs` to highlight.
    fill_value
        Fill value for observations not present in selected batches
    scale
        Whether to linearly scale the distribution.
    dot_scale_factor
        Scale factor for the ``time_points``.
    cmap
        Colormap for continuous observations.
    na_color
        Color for NaN values.
    title
        Title of the figure.
    suptitle
        Suptitle of the figure.
    suptitle_fontsize
        Font size of the suptitle.
    ax
        Ax used for plotting. If :obj:`None`, create a new one.
    return_fig
        Whether to return the figure.
    figsize
        Size of the figure.
    dpi
        Dots per inch.
    save
        Path where to save the figure.
    kwargs
        Keyword arguments for :func:`~scanpy.pl.embedding`.

    Returns
    -------
    If ``return_fig = True``, returns and plots the figure. Otherwise, just plots the figure.
    """
    adata, _ = _input_to_adatas(obj)
    if key not in adata.obs:
        raise KeyError(f"No data found in `adata.obs[{key!r}]`.")

    data = get_plotting_vars(adata, _constants.PULL, key=key)
    if data["data"] is not None and data["subset"] is not None and cmap is None:
        cmap = _create_col_colors(adata, data["data"], data["subset"])

    fig = _plot_temporal(
        adata=adata,
        temporal_key=data["temporal_key"],
        key_stored=key,
        source=data["source"],
        target=data["target"],
        categories=data["subset"],
        push=False,
        time_points=time_points,
        basis=basis,
        constant_fill_value=fill_value,
        scale=scale,
        save=save,
        cont_cmap=cmap,
        dot_scale_factor=dot_scale_factor,
        na_color=na_color,
        title=title,
        suptitle=suptitle,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        suptitle_fontsize=suptitle_fontsize,
        **kwargs,
    )
    return fig if return_fig else None

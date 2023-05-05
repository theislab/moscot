from types import MappingProxyType
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
from moscot._docs._docs_plot import d_plotting
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


@d_plotting.dedent
def cell_transition(
    inp: Union[AnnData, Tuple[AnnData, AnnData], "CompoundProblem"],  # type: ignore[type-arg]
    uns_key: str = _constants.CELL_TRANSITION,
    row_labels: Optional[str] = None,
    col_labels: Optional[str] = None,
    annotate: Optional[str] = "{x:.2f}",
    fontsize: float = 7.0,
    cmap: Union[str, mpl.colors.Colormap] = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[str] = None,
    ax: Optional[mpl.axes.Axes] = None,
    return_fig: bool = False,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Plot a cell transition matrix.

    %(desc_cell_transition)s

    Parameters
    ----------
    %(input_plotting)s
    %(uns_key)s
    %(transition_labels_cell_transition)s
    %(fontsize)s
    %(cmap)s
    %(figsize_dpi_save)s
    %(cbar_kwargs_cell_transition)s

    Returns
    -------
    %(return_cell_transition)s

    Notes
    -----
    %(notes_cell_transition)s
    """
    adata1, adata2 = _input_to_adatas(inp)
    data = get_plotting_vars(adata1, _constants.CELL_TRANSITION, key=uns_key)

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
        row_annotation_label=data["source"] if row_labels is None else row_labels,
        col_annotation_label=data["target"] if col_labels is None else col_labels,
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


@d_plotting.dedent
def sankey(
    inp: Union[AnnData, "TemporalProblem", "LineageProblem"],
    # TODO(MUCDK): rename to `key` or `key_added`?
    uns_key: str = _constants.SANKEY,
    captions: Optional[List[str]] = None,
    title: Optional[str] = None,
    colors_dict: Optional[Dict[str, float]] = None,
    alpha: float = 1.0,
    interpolate_color: bool = False,
    cmap: Union[str, mpl.colors.Colormap] = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[str] = None,
    ax: Optional[mpl.axes.Axes] = None,
    return_fig: bool = False,
    **kwargs: Any,
) -> Optional[mpl.figure.Figure]:
    """
    Plot a Sankey diagram.

    {desc_sankey}

    Parameters
    ----------
    %(input_plotting)s
    %(uns_key)s
    %(captions_sankey)s
    %(title)s
    %(colors_dict_sankey)s
    %(alpha_transparency)s
    %(interpolate_color)s
    %(cmap)s
    %(figsize_dpi_save)s
    %(sankey_kwargs)s

    Returns
    -------
    %(return_sankey)s

    Notes
    -----
    %(notes_sankey)s
    """
    adata, _ = _input_to_adatas(inp)
    data = get_plotting_vars(adata, _constants.SANKEY, key=uns_key)

    fig = _sankey(
        adata=adata,
        key=data["key"],
        transition_matrices=data["transition_matrices"],
        captions=data["captions"] if captions is None else captions,
        colorDict=colors_dict,
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


@d_plotting.dedent
def push(
    inp: Union[AnnData, "TemporalProblem", "LineageProblem", "SpatioTemporalProblem", "CompoundProblem"],  # type: ignore[type-arg]  # noqa: 501
    uns_key: str = _constants.PUSH,
    time_points: Optional[Sequence[float]] = None,
    basis: str = "umap",
    fill_value: float = np.nan,
    scale: bool = True,
    title: Optional[Union[str, List[str]]] = None,
    suptitle: Optional[str] = None,
    cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
    dot_scale_factor: float = 2.0,
    na_color: str = "#e8ebe9",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[str] = None,
    ax: Optional[mpl.axes.Axes] = None,
    return_fig: bool = False,
    suptitle_fontsize: Optional[float] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Visualise the push result in an embedding.

    %(desc_sankey)s

    Parameters
    ----------
    %(input_plotting)s
    %(uns_key)s
    %(time_points_push_pull)s
    %(basis_push_pull)s
    %(fill_value_push_pull)s
    %(scale_push_pull)s
    %(title)s
    %(cmap)s
    %(dot_scale_factor)s
    %(na_color)s
    %(figsize_dpi_save)s
    %(suptitle_fontsize)s

    Returns
    -------
    %(return_push_pull)s

    Notes
    -----
    %(return_push_pull)s
    """
    adata, _ = _input_to_adatas(inp)
    if uns_key not in adata.obs:
        raise KeyError(f"No data found in `adata.obs[{uns_key!r}]`.")

    data = get_plotting_vars(adata, _constants.PUSH, key=uns_key)
    if data["data"] is not None and data["subset"] is not None and cmap is None:
        cmap = _create_col_colors(adata, data["data"], data["subset"])

    fig = _plot_temporal(
        adata=adata,
        temporal_key=data["temporal_key"],
        key_stored=uns_key,
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


@d_plotting.dedent
def pull(
    inp: Union[AnnData, "TemporalProblem", "LineageProblem", "SpatioTemporalProblem", "CompoundProblem"],  # type: ignore[type-arg]  # noqa: 501
    uns_key: str = _constants.PULL,
    time_points: Optional[Sequence[float]] = None,
    basis: str = "umap",
    fill_value: float = np.nan,
    scale: bool = True,
    title: Optional[Union[str, List[str]]] = None,
    suptitle: Optional[str] = None,
    cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
    dot_scale_factor: float = 2.0,
    na_color: str = "#e8ebe9",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[str] = None,
    ax: Optional[mpl.axes.Axes] = None,
    return_fig: bool = False,
    suptitle_fontsize: Optional[float] = None,
    **kwargs: Any,
) -> Optional[mpl.figure.Figure]:
    """
    Visualise the pull result in an embedding.

    %(desc_sankey)s

    Parameters
    ----------
    %(input_plotting)s
    %(uns_key)s
    %(time_points_push_pull)s
    %(basis_push_pull)s
    %(fill_value_push_pull)s
    %(scale_push_pull)s
    %(title)s
    %(cmap)s
    %(dot_scale_factor)s
    %(na_color)s
    %(figsize_dpi_save)s
    %(suptitle_fontsize)s

    Returns
    -------
    %(return_push_pull)s

    Notes
    -----
    %(return_push_pull)s
    """
    adata, _ = _input_to_adatas(inp)

    if uns_key not in adata.obs:
        raise KeyError(f"No data found in `adata.obs[{uns_key!r}]`.")

    data = get_plotting_vars(adata, _constants.PULL, key=uns_key)
    if data["data"] is not None and data["subset"] is not None and cmap is None:
        cmap = _create_col_colors(adata, data["data"], data["subset"])

    fig = _plot_temporal(
        adata=adata,
        temporal_key=data["temporal_key"],
        key_stored=uns_key,
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

from types import MappingProxyType
from typing import Any, Dict, List, Type, Tuple, Union, Mapping, Iterable, Optional

from matplotlib import colors as mcolors
from matplotlib.axes import Axes
import matplotlib as mpl

import numpy as np

from anndata import AnnData

from moscot._docs import d
from moscot.pl._utils import _sankey, _heatmap, _plot_temporal, _input_to_adatas
from moscot.problems.base import CompoundProblem  # type: ignore[attr-defined]
from moscot.problems.time import LineageProblem, TemporalProblem  # type: ignore[attr-defined]
from moscot._constants._constants import AdataKeys, PlottingKeys, PlottingDefaults
from moscot.problems.base._compound_problem import K


@d.dedent
def cell_transition(
    input: Union[AnnData, Tuple[AnnData, AnnData], Type[CompoundProblem]],
    key_stored: Optional[str] = None,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    row_annotation_label: Optional[str] = None,
    col_annotation_label: Optional[str] = None,
    annotate_values: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[bool] = False,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Plot a cell transition matrix.

    In order to run this function the corresponding method `cell_transition` of the :class:`moscot.problems` instance
    must have been run, see `NOTES` for requirements.

    Parameters
    ----------
    %(input_plotting)s
    %(key_stored)s
    %(cont_cmap)s
    row_annotation_label
        Whether to add annotations to the rows of the transition matrix
    col_annotation_label
        Whether to add annotations to the columns of the transition matrix
    annotate_values
        Whether to add the values to the entries of the transition matrix
    %(plotting)s
    %(cbar_kwargs)s
    %(ax)s
    %(return_fig)s
    kwargs
        key word arguments for TODO

    Returns
    -------
    A cell transition figure, an instance of :class:`matplotlib.figure.Figure`.

    Notes
    -----
    This function looks for the following data in the :class:`anndata.AnnData` object which is passed or saved
    as an attribute of the :mod:`moscot.problems` instance.

    - transition_matrix
    - source_annotation
    - target_annotation
    - source_key
    - source_target

    """
    adata1, adata2 = _input_to_adatas(input)

    key = PlottingDefaults.CELL_TRANSITION if key_stored is None else key_stored
    if key not in adata1.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION]:
        raise KeyError("TODO.")
    data = adata1.uns[AdataKeys.UNS][PlottingKeys.CELL_TRANSITION][key]

    return _heatmap(
        row_adata=adata1,
        col_adata=adata2,
        transition_matrix=data["transition_matrix"],
        row_annotation=data["source_annotation"],
        col_annotation=data["target_annotation"],
        row_annotation_label=data["source_key"] if row_annotation_label is None else row_annotation_label,
        col_annotation_label=data["target_key"] if col_annotation_label is None else col_annotation_label,
        cont_cmap=cont_cmap,
        annotate_values=annotate_values,
        figsize=figsize,
        dpi=dpi,
        cbar_kwargs=cbar_kwargs,
        ax=ax,
        save=save,
        return_fig=return_fig,
        **kwargs,
    )


@d.dedent
def sankey(
    input: Union[AnnData, TemporalProblem, LineageProblem],
    key_stored: Optional[str] = None,
    captions: Optional[List[str]] = None,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    colorDict: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[bool] = False,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Plot a sankey diagram.

    In order to run this function the corresponding method `sankey` of the :mod:`moscot.problems` instance
    must have been run, see `NOTES` for requirements.

    Parameters
    ----------
    input
        An instance of :class:`anndata.AnnData` where the corresponding information obtained by the `sankey` method
        of the :mod:`moscot.problems` instance is saved. Alternatively, the problem instance can be passed directly, i.e. an
        instance of :class:`moscot.problems.time.TemporalProblem` or :class:`moscot.problems.time.LineageProblem`.
    %(key_stored)s
    %(cont_cmap)s
    %(plotting)s
    %(cbar_kwargs)s
    %(ax)s
    %(return_fig)s
    kwargs
        key word arguments for TODO

    Returns
    -------
    A sankey figure, an instance of :class:`matplotlib.figure.Figure`.

    Notes
    -----
    This function looks for the following data in the :class:`anndata.AnnData` object which is passed or saved
    as an attribute of the :mod:`moscot.problems` instance.

    - `transition_matrices`
    - `captions``
    - `key`

    """
    adata, _ = _input_to_adatas(input)

    key = PlottingDefaults.SANKEY if key_stored is None else key_stored
    if key not in adata.uns[AdataKeys.UNS][PlottingKeys.SANKEY]:
        raise KeyError("TODO.")
    data = adata.uns[AdataKeys.UNS][PlottingKeys.SANKEY][key]
    fig = _sankey(
        adata=adata,
        key=data["key"],
        transition_matrices=data["transition_matrices"],
        captions=data["captions"] if captions is None else captions,
        colorDict=colorDict,
        cont_cmap=cont_cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )
    if save:
        fig.save(save)
    return fig


@d.dedent
def push(
    input: Union[AnnData, TemporalProblem, LineageProblem],
    key_stored: Optional[str] = None,
    time_points: Optional[Iterable[K]] = None,
    basis: str = "umap",
    result_key: str = "plot_tmp",
    constant_fill_value: float = np.nan,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[bool] = False,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Visualise the push result in an embedding.

    Parameters
    ----------
    input
        An instance of :class:`anndata.AnnData` where the corresponding information obtained by the `sankey` method
        of the :mod:`moscot.problems` instance is saved. Alternatively, the problem instance can be passed directly, i.e. an
        instance of :class:`moscot.problems.time.TemporalProblem` or :class:`moscot.problems.time.LineageProblem`.
    %(key_stored)s
    %(plot_time_points)s
    basis
        basis of the embedding, saved in `adata.obsm[x_{[basis]}]`.
    result_key
        column of :attr:`anndata.AnnData.obs` where the result is stored
    %(constant_fill_value)s
    %(cont_cmap)s
    %(plotting)s
    %(ax)s
    %(return_fig)s

    kwargs
        keyword arguments for :func:`scanpy.pl.scatter`.


    Returns
    -------
    A figure visualising a push distribution, an instance of :class:`matplotlib.figure.Figure`.

    Notes
    -----
    This function looks for the following data in the :class:`anndata.AnnData` object which is passed or saved
    as an attribute of the :mod:`moscot.problems` instance.

    - `temporal_key`
    """
    adata, _ = _input_to_adatas(input)

    key = PlottingDefaults.PUSH if key_stored is None else key_stored
    if key not in adata.obs:
        raise KeyError("TODO.")
    data = adata.uns[AdataKeys.UNS][PlottingKeys.PUSH][key]
    _plot_temporal(
        adata=adata,
        temporal_key=data["temporal_key"],
        key_stored=key,
        time_points=time_points,
        basis=basis,
        result_key=result_key,
        constant_fill_value=constant_fill_value,
        save=save,
        cont_cmap=cont_cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )


@d.dedent
def pull(
    input: Union[AnnData, TemporalProblem, LineageProblem],
    key_stored: Optional[str] = None,
    time_points: Optional[Iterable[K]] = None,
    basis: str = "umap",
    result_key: str = "plot_tmp",
    constant_fill_value: float = np.nan,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[bool] = False,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Visualise the push result in an embedding.

    Parameters
    ----------
    input
        An instance of :class:`anndata.AnnData` where the corresponding information obtained by the `sankey` method
        of the :mod:`moscot.problems` instance is saved. Alternatively, the problem instance can be passed directly, i.e. an
        instance of :class:`moscot.problems.time.TemporalProblem` or :class:`moscot.problems.time.LineageProblem`.
    %(key_stored)s
    %(plot_time_points)s
    result_key
        column of :attr:`anndata.AnnData.obs` where the result is stored
    %(constant_fill_value)s
    %(cont_cmap)s
    %(plotting)s
    %(ax)s
    %(return_fig)s

    kwargs
        keyword arguments for :func:`scanpy.pl.scatter`.


    Returns
    -------
    A figure visualising a push distribution, an instance of :class:`matplotlib.figure.Figure`.

    Notes
    -----
    This function looks for the following data in the :class:`anndata.AnnData` object which is passed or saved
    as an attribute of the :mod:`moscot.problems` instance.

    - `temporal_key`

    """
    adata, _ = _input_to_adatas(input)

    key = PlottingDefaults.PULL if key_stored is None else key_stored
    if key not in adata.obs:
        raise KeyError("TODO.")
    data = adata.uns[AdataKeys.UNS][PlottingKeys.PULL][key]
    _plot_temporal(
        adata=adata,
        temporal_key=data["temporal_key"],
        key_stored=key,
        time_points=time_points,
        basis=basis,
        result_key=result_key,
        constant_fill_value=constant_fill_value,
        save=save,
        cont_cmap=cont_cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        return_fig=return_fig,
        **kwargs,
    )

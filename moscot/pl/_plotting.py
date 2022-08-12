from types import MappingProxyType
from typing import Any, List, Type, Tuple, Union, Mapping, Optional

from matplotlib import colors as mcolors
from matplotlib.axes import Axes
import matplotlib as mpl

from anndata import AnnData

from moscot._docs import d
from moscot.pl._utils import _sankey, _heatmap, _input_to_adatas
from moscot.problems.base import CompoundProblem  # type: ignore[attr-defined]
from moscot.problems.time import LineageProblem, TemporalProblem  # type: ignore[attr-defined]
from moscot._constants._constants import AdataKeys, PlottingKeys, PlottingDefaults


@d.dedent
def cell_transition(
    input: Union[AnnData, Tuple[AnnData, AnnData], Type[CompoundProblem]],
    key_stored: Optional[str] = None,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    annotate_rows: bool = True,
    annotate_cols: bool = True,
    annotate_values: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[bool] = False,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Optional[Axes] = None,
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
    annotate_rows
        Whether to add annotations to the rows of the transition matrix
    annotate_cols
        Whether to add annotations to the columns of the transition matrix
    annotate_values
        Whether to add the values to the entries of the transition matrix
    %(plotting)s
    %(cbar_kwargs)s
    %(ax)s
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
        row_annotation_suffix=data["source_key"],
        col_annotation_suffix=data["source_target"],
        cont_cmap=cont_cmap,
        annotate_rows=annotate_rows,
        annotate_cols=annotate_cols,
        annotate_values=annotate_values,
        figsize=figsize,
        dpi=dpi,
        cbar_kwargs=cbar_kwargs,
        ax=ax,
        **kwargs,
    )


@d.dedent
def sankey(
    input: Union[AnnData, TemporalProblem, LineageProblem],
    key_stored: Optional[str] = None,
    captions: Optional[List[str]] = None,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[bool] = False,
    ax: Optional[Axes] = None,
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
    kwargs
        key word arguments for TODO

    Returns
    -------
    A cell transition figure, an instance of :class:`matplotlib.figure.Figure`.

    Notes
    -----
    This function looks for the following data in the :class:`anndata.AnnData` object which is passed or saved
    as an attribute of the :mod:`moscot.problems` instance.

    - transition_matrices
    - captions
    - key

    """
    adata = _input_to_adatas(input)

    key = PlottingDefaults.CELL_TRANSITION if key_stored is None else key_stored
    if key not in adata.uns[AdataKeys.UNS][PlottingKeys.SANKEY]:  # type: ignore[attr-defined]
        raise KeyError("TODO.")
    data = adata.uns[AdataKeys.UNS][PlottingKeys.SANKEY][key]  # type: ignore[attr-defined]
    fig = _sankey(
        adata=adata,
        key=data["key"],
        transition_matrices=data["transition_matrices"],
        captions=data["captions"] if captions is None else captions,
        colorDict=cont_cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        **kwargs,
    )
    if save:
        fig.save(save)
    return fig

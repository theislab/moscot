from types import MappingProxyType
from typing import Any, Dict, List, Type, Tuple, Union, Mapping, Optional

from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import matplotlib as mpl

from anndata import AnnData

from moscot.pl._utils import _sankey, _heatmap, _input_to_adatas
from moscot.problems.base import CompoundProblem  # type: ignore[attr-defined]
from moscot._constants._constants import AdataKeys, PlottingKeys, PlottingDefaults


def cell_transition(
    input: Union[AnnData, Tuple[AnnData, AnnData], Type[CompoundProblem]],
    key_stored: Optional[str] = None,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    annotate_rows: bool = True,
    annotate_cols: bool = True,
    annotate_values: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Plot cell transition matrix.

    TODO.
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


def sankey(
    input: Union[AnnData, Tuple[AnnData, AnnData], Type[CompoundProblem]],
    key_stored: Optional[str] = None,
    captions: Optional[List[str]] = None,
    colorDict: Optional[Union[Dict[Any, str], ListedColormap]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    """
    Plot cell transition matrix.

    TODO.
    """
    adata = _input_to_adatas(input)

    key = PlottingDefaults.CELL_TRANSITION if key_stored is None else key_stored
    if key not in adata.uns[AdataKeys.UNS][PlottingKeys.SANKEY]:  # type: ignore[attr-defined]
        raise KeyError("TODO.")
    data = adata.uns[AdataKeys.UNS][PlottingKeys.SANKEY][key]  # type: ignore[attr-defined]
    _sankey(
        adata=adata,
        key=key,
        transition_matrices=data["transition_matrices"],
        captions=data["captions"] if captions is None else captions,
        colorDict=colorDict,
        title=title,
        figsize=figsize,
        dpi=dpi,
        ax=ax,
        **kwargs,
    )

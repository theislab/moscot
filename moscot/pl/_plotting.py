from copy import copy
from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Union, Mapping, Optional, TYPE_CHECKING, Type
from pathlib import Path
from collections import defaultdict
import os
from anndata import AnnData

from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib as mpl

import numpy as np

from scanpy import logging as logg, settings
from anndata import AnnData
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation as add_color_palette

from moscot.problems.base import CompoundProblem
from moscot._docs import d
from moscot.pl._utils import _sankey, _heatmap


def cell_transition(adata: Union[AnnData, Tuple[AnnData, AnnData]], key: Optional[str] = None, row_annotation_suffix: Optional[str] = None, col_annotation_suffix: Optional[str] = None,  cont_cmap: Union[str, mcolors.Colormap] = "viridis", annotate_rows: bool=True, annotate_cols: bool=True, annotate_values: bool = True, figsize: Optional[Tuple[float, float]] =None, dpi: Optional[int] = None, cbar_kwargs: Mapping[str, Any] = MappingProxyType({}), ax: Optional[Axes] = None , **kwargs:Any) -> mpl.figure.Figure:
    if isinstance(adata, AnnData):
        adata_2 = adata
    else:
        adata, adata_2 = adata

    key = "cell_transition" if key is None else key
    if key not in adata.uns["moscot_results"]["cell_transition"]:
        raise KeyError("TODO.")
    data = adata.uns["moscot_results"]["cell_transition"][key]

    return _heatmap(
    row_adata = adata,
    col_adata = adata_2,
    transition_matrix =  data["transition_matrix"],
    row_annotation =  data["row_annotation"],
    col_annotation = data["col_annotation"],
    row_annotation_suffix = row_annotation_suffix,
    col_annotation_suffix = col_annotation_suffix,
    cont_cmap = cont_cmap,
    annotate_rows=annotate_rows,
    annotate_cols=annotate_cols,
    annotate_values=annotate_values,
    figsize=figsize,
    dpi=dpi,
    cbar_kwargs=cbar_kwargs,
    ax=ax,
    **kwargs)






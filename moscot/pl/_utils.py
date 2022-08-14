from copy import copy
from types import MappingProxyType
from typing import Any, Set, Dict, List, Tuple, Union, Mapping, Iterable, Optional, TYPE_CHECKING
from pathlib import Path
from collections import defaultdict
import os

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
import scanpy as sc

from moscot._docs import d
from moscot.problems.base import CompoundProblem  # type: ignore[attr-defined]
from moscot.problems.base._compound_problem import K


@d.dedent
def save_fig(fig: Figure, path: Union[str, Path], make_dir: bool = True, ext: str = "png", **kwargs: Any) -> None:
    """
    Save a figure.

    Parameters
    ----------
    fig
        Figure to save.
    path
        Path where to save the figure. If path is relative, save it under :attr:`scanpy.settings.figdir`.
    make_dir
        Whether to try making the directory if it does not exist.
    ext
        Extension to use if none is provided.
    kwargs
        Keyword arguments for :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    None
        Just saves the plot.
    """
    if os.path.splitext(path)[1] == "":
        path = f"{path}.{ext}"

    path = Path(path)

    if not path.is_absolute():
        path = Path(settings.figdir) / path

    if make_dir:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logg.debug(f"Unable to create directory `{path.parent}`. Reason: `{e}`")

    logg.debug(f"Saving figure to `{path!r}`")

    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("transparent", True)

    fig.savefig(path, **kwargs)


def set_palette(
    adata: AnnData,
    key: str,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    force_update_colors: bool = False,
    **_: Any,
) -> None:
    """Set palette."""
    if key not in adata.obs.columns:
        raise KeyError("TODO: invalid key.")
    uns_key = f"{key}_colors"
    if uns_key not in adata.uns:
        add_color_palette(adata, key=key, palette=cont_cmap, force_update_colors=force_update_colors)


def _sankey(
    adata: AnnData,
    key: str,
    transition_matrices: List[pd.DataFrame],
    captions: Optional[List[str]] = None,
    colorDict: Optional[Union[Dict[Any, str], ListedColormap]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    ax: Optional[Axes] = None,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    fontsize: float = 12.0,
    horizontal_space: float = 1.5,
    force_update_colors: bool = False,
    **kwargs: Any,
) -> mpl.figure.Figure:
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, dpi=dpi, figsize=figsize)
    else:
        ax.figure
    if captions is not None and len(captions) != len(transition_matrices):
        raise ValueError("TODO: If `captions` are specified length has to be same as of `transition_matrices`.")
    if colorDict is None:
        # TODO: adapt for unique categories
        set_palette(adata=adata, key=key, cont_cmap=cont_cmap, force_update_colors=force_update_colors)

        colorDict = {cat: adata.uns[f"{key}_colors"][i] for i, cat in enumerate(adata.obs[key].cat.categories)}
    else:
        missing = [label for label in adata.obs[key].cat.categories if label not in colorDict.keys()]
        if missing:
            msg = "The colorDict parameter is missing values for the following labels : "
            msg += "{}".format(", ".join(missing))
            raise ValueError(msg)
    left_pos = [0]
    for ind, dataFrame in enumerate(transition_matrices):
        dataFrame /= dataFrame.sum().sum()
        leftLabels = list(dataFrame.index)
        rightLabels = list(dataFrame.columns)
        set(leftLabels).union(set(rightLabels))

        # Determine positions of left label patches and total widths
        leftWidths: Dict[Any, Dict[Any, float]] = defaultdict()
        for i, leftLabel in enumerate(leftLabels):
            myD = {}
            myD["left"] = dataFrame.loc[leftLabel, :].sum()
            if i == 0:
                myD["bottom"] = 0
                myD["top"] = myD["left"]
            else:
                myD["bottom"] = leftWidths[leftLabels[i - 1]]["top"]
                myD["top"] = myD["bottom"] + myD["left"]
                topEdge = myD["top"]
            leftWidths[leftLabel] = myD

        # Determine positions of right label patches and total widths
        rightWidths: Dict[Any, Dict[Any, float]] = defaultdict()
        for i, rightLabel in enumerate(rightLabels):
            myD = {}
            myD["right"] = dataFrame.loc[:, rightLabel].sum()
            if i == 0:
                myD["bottom"] = 0
                myD["top"] = myD["right"]
            else:
                myD["bottom"] = rightWidths[rightLabels[i - 1]]["top"]
                myD["top"] = myD["bottom"] + myD["right"]
                topEdge = myD["top"]
            rightWidths[rightLabel] = myD

        # Total vertical extent of diagram
        xMax = topEdge

        # Draw vertical bars on left and right of each label"s section & print label
        for leftLabel in leftLabels:
            if ind == 0:
                plt.fill_between(
                    [-0.02 * xMax, 0],
                    2 * [leftWidths[leftLabel]["bottom"]],
                    2 * [leftWidths[leftLabel]["bottom"] + leftWidths[leftLabel]["left"]],
                    color=colorDict[leftLabel],
                    alpha=0.99,
                )
                plt.text(
                    -0.05 * xMax,
                    leftWidths[leftLabel]["bottom"] + 0.5 * leftWidths[leftLabel]["left"],
                    leftLabel,
                    {"ha": "right", "va": "center"},
                    fontsize=fontsize,
                )
        for rightLabel in rightLabels:
            plt.fill_between(
                [xMax + left_pos[ind], 1.02 * xMax + left_pos[ind]],
                2 * [rightWidths[rightLabel]["bottom"]],
                2 * [rightWidths[rightLabel]["bottom"] + rightWidths[rightLabel]["right"]],
                color=colorDict[rightLabel],
                alpha=0.99,
            )
            plt.text(
                1.05 * xMax + left_pos[ind],
                rightWidths[rightLabel]["bottom"] + 0.5 * rightWidths[rightLabel]["right"],
                rightLabel,
                {"ha": "left", "va": "center"},
                fontsize=fontsize,
            )
        np.min([leftWidths[cat]["bottom"] for cat in leftWidths.keys()])

        if captions is not None:
            plt.text(left_pos[ind] + 0.3 * xMax, -0.1, captions[ind])

        left_pos += [horizontal_space * xMax]

        # Plot strips
        for leftLabel in leftLabels:
            for rightLabel in rightLabels:
                labelColor = leftLabel
                if dataFrame.loc[leftLabel, rightLabel] > 0:
                    # Create array of y values for each strip, half at left value,
                    # half at right, convolve
                    ys_d = np.array(50 * [leftWidths[leftLabel]["bottom"]] + 50 * [rightWidths[rightLabel]["bottom"]])
                    ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
                    ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
                    ys_u = np.array(
                        50 * [leftWidths[leftLabel]["bottom"] + dataFrame.loc[leftLabel, rightLabel]]
                        + 50 * [rightWidths[rightLabel]["bottom"] + dataFrame.loc[leftLabel, rightLabel]]
                    )
                    ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")
                    ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")

                    # Update bottom edges at each label so next strip starts at the right place
                    leftWidths[leftLabel]["bottom"] += dataFrame.loc[leftLabel, rightLabel]
                    rightWidths[rightLabel]["bottom"] += dataFrame.loc[leftLabel, rightLabel]

                    if ind == 0:
                        plt.fill_between(
                            np.linspace(0 + left_pos[ind], xMax + left_pos[ind], len(ys_d)),
                            ys_d,
                            ys_u,
                            alpha=0.65,
                            color=colorDict[labelColor],
                        )
                    else:
                        plt.fill_between(
                            np.linspace(0 + left_pos[ind], xMax + left_pos[ind], len(ys_d)),
                            ys_d,
                            ys_u,
                            alpha=0.65,
                            color=colorDict[labelColor],
                        )

        plt.gca().axis("off")
        plt.gcf().set_size_inches(6, 6)
        if title is not None:
            plt.title(title)


def _heatmap(
    row_adata: AnnData,
    col_adata: AnnData,
    transition_matrix: pd.DataFrame,
    row_annotation: str,
    col_annotation: str,
    row_annotation_suffix: str = "",
    col_annotation_suffix: str = "",
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

    cbar_kwargs = dict(cbar_kwargs)

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, dpi=dpi, figsize=figsize)
    else:
        fig = ax.figure

    row_cmap, col_cmap, row_norm, col_norm = _get_cmap_norm(
        row_adata, col_adata, transition_matrix, row_annotation, col_annotation
    )

    row_sm = mpl.cm.ScalarMappable(cmap=row_cmap, norm=row_norm)
    col_sm = mpl.cm.ScalarMappable(cmap=col_cmap, norm=col_norm)

    norm = mpl.colors.Normalize(
        vmin=kwargs.pop("vmin", np.nanmin(transition_matrix)), vmax=kwargs.pop("vmax", np.nanmax(transition_matrix))
    )
    cont_cmap = copy(plt.get_cmap(cont_cmap))
    cont_cmap.set_bad(color="grey")

    im = ax.imshow(transition_matrix[::-1], cmap=cont_cmap, norm=norm)
    ax.grid(False)
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])

    if annotate_values:
        _annotate_heatmap(transition_matrix, im, cmap=cont_cmap, **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)

    _ = fig.colorbar(
        im,
        cax=cax,
        ticks=np.linspace(norm.vmin, norm.vmax, 10),
        orientation="vertical",
        format="%0.2f",
        **cbar_kwargs,
    )

    col_cats = divider.append_axes("top", size="2%", pad=0)
    c = fig.colorbar(col_sm, cax=col_cats, orientation="horizontal", ticklocation="top" if annotate_cols else "auto")
    if annotate_cols:
        c.set_ticks(np.arange(transition_matrix.shape[1]) + 0.5)
        c.ax.set_xticklabels(transition_matrix.columns, rotation=90)
        c.set_label(col_annotation + col_annotation_suffix)

    row_cats = divider.append_axes("left", size="2%", pad=0)
    c = fig.colorbar(row_sm, cax=row_cats, orientation="vertical", ticklocation="left" if annotate_rows else "auto")
    if annotate_rows:
        c.set_ticks(np.arange(transition_matrix.shape[0]) + 0.5)
        c.ax.set_yticklabels(transition_matrix.index)
        c.set_label(row_annotation + row_annotation_suffix)

    return fig


def _get_black_or_white(value: float, cmap: mcolors.Colormap) -> str:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Value must be in range `[0, 1]`, found `{value}`.")

    r, g, b, *_ = (int(c * 255) for c in cmap(value))
    return _contrasting_color(r, g, b)


def _annotate_heatmap(
    transition_matrix: pd.DataFrame,
    im: mpl.image.AxesImage,
    valfmt: str = "{x:.2f}",
    cmap: Union[mpl.colors.Colormap, str] = "viridis",
    **kwargs: Any,
) -> None:
    # modified from matplotlib's site
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    kw = {"ha": "center", "va": "center"}
    kw.update(**kwargs)

    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)
    if TYPE_CHECKING:
        assert callable(valfmt)

    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            val = im.norm(transition_matrix.iloc[transition_matrix.shape[0] - (i + 1), j])
            if np.isnan(val):
                continue
            kw.update(color=_get_black_or_white(val, cmap))
            im.axes.text(j, i, valfmt(transition_matrix.iloc[transition_matrix.shape[0] - (i + 1), j], None), **kw)


def _get_cmap_norm(
    row_adata: AnnData,
    col_adata: AnnData,
    transition_matrix: pd.DataFrame,
    row_annotation: str,
    col_annotation: str,
) -> Tuple[mcolors.ListedColormap, mcolors.ListedColormap, mcolors.BoundaryNorm, mcolors.BoundaryNorm]:
    row_color_dict = {
        row_adata.obs[row_annotation].cat.categories[i]: col
        for i, col in enumerate(row_adata.uns[f"{row_annotation}_colors"])
    }
    col_color_dict = {
        col_adata.obs[col_annotation].cat.categories[i]: col
        for i, col in enumerate(col_adata.uns[f"{col_annotation}_colors"])
    }

    row_colors = [row_color_dict[cat] for cat in transition_matrix.index]  # [::-1]
    col_colors = [col_color_dict[cat] for cat in transition_matrix.columns]

    row_cmap = mcolors.ListedColormap(row_colors)
    col_cmap = mcolors.ListedColormap(col_colors)
    row_norm = mcolors.BoundaryNorm(np.arange(transition_matrix.shape[0] + 1), transition_matrix.shape[0])
    col_norm = mcolors.BoundaryNorm(np.arange(transition_matrix.shape[1] + 1), transition_matrix.shape[1])

    return row_cmap, col_cmap, row_norm, col_norm


def _contrasting_color(r: int, g: int, b: int) -> str:
    for val in [r, g, b]:
        assert 0 <= val <= 255, f"Color value `{val}` is not in `[0, 255]`."

    return "#000000" if r * 0.299 + g * 0.587 + b * 0.114 > 186 else "#ffffff"


def _input_to_adatas(input: Union[AnnData, Tuple[AnnData, AnnData]]) -> Tuple[AnnData, AnnData]:
    if isinstance(input, CompoundProblem):
        return input.adata, input._other_adata if hasattr(input, "_other_adata") else input.adata
    else:
        if isinstance(input, AnnData):
            return input, input
        elif isinstance(input, tuple):
            if not isinstance(input[0], AnnData):
                raise TypeError("TODO: input must be `AnnData`.")
            if not isinstance(input[1], AnnData):
                raise TypeError("TODO: input must be `AnnData`.")
            return input  # type: ignore[return-value]
        else:
            raise NotImplementedError("TODO.")


def _plot_temporal(
    adata: AnnData,
    temporal_key: str,
    key_stored: str,
    plot_time_points: Optional[Iterable[K]] = None,
    basis: str = "umap",
    result_key: str = "plot_tmp",
    fill_keys: Iterable[K] = [],
    fill_value: float = 0.0,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[bool] = False,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> None:
    all_keys = adata.obs[temporal_key].unique()
    if plot_time_points is None:
        fill_keys: Set[K] = set()
    else:
        fill_keys = set(all_keys) - set(plot_time_points)
    tmp = np.full(len(adata), np.nan)
    for t in adata.obs[temporal_key].unique():
        mask = adata.obs[temporal_key] == t
        if t in fill_keys:
            tmp[mask] = fill_value
        else:
            tmp[mask] = adata[adata.obs[temporal_key] == t].obs[key_stored]

    adata.obs[result_key] = tmp

    sc.set_figure_params(figsize=figsize, dpi=dpi)  # TODO(@MUCDK, michalk8): necessary? want to make it uniform
    sc.pl.embedding(
        adata=adata, basis=basis, color=result_key, color_map=cont_cmap, title=title, save=save, ax=ax, **kwargs
    )

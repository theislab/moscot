from copy import copy
from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Union, Mapping, Optional, Sequence, TYPE_CHECKING
from collections import defaultdict

from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.axes import Axes
from pandas.api.types import is_categorical_dtype
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib as mpl

import numpy as np

from anndata import AnnData
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation as add_color_palette
import scanpy as sc

from moscot.problems.base import CompoundProblem  # type: ignore[attr-defined]
from moscot._constants._constants import AggregationMode
from moscot.problems.base._compound_problem import K


def set_palette(
    adata: AnnData,
    key: str,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    force_update_colors: bool = False,
    **_: Any,
) -> None:
    """Set palette."""
    if key not in adata.obs:
        raise KeyError(f"Unable to find data in `adata.obs[{key!r}]`.")
    uns_key = f"{key}_colors"
    if uns_key not in adata.uns:
        add_color_palette(adata, key=key, palette=cont_cmap, force_update_colors=force_update_colors)


# adapted from https://github.com/anazalea/pySankey/blob/master/pysankey/sankey.py
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
    alpha: float = 1.0,
    interpolate_color: bool = False,
    **kwargs: Any,
) -> mpl.figure.Figure:
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, dpi=dpi, figsize=figsize)
    if captions is not None and len(captions) != len(transition_matrices):
        raise ValueError(f"Expected captions to be of length `{len(transition_matrices)}`, found `{len(captions)}`.")
    if colorDict is None:
        # TODO: adapt for unique categories
        set_palette(adata=adata, key=key, cont_cmap=cont_cmap, force_update_colors=force_update_colors)

        colorDict = {cat: adata.uns[f"{key}_colors"][i] for i, cat in enumerate(adata.obs[key].cat.categories)}
    else:
        missing = sorted(label for label in adata.obs[key].cat.categories if label not in colorDict)
        if missing:
            raise ValueError(f"The following labels have missing colors: `{missing}`.")
    left_pos = [0]
    for ind, dataFrame in enumerate(transition_matrices):
        dataFrame /= dataFrame.values.sum()
        leftLabels = list(dataFrame.index)
        rightLabels = list(dataFrame.columns)

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
                ax.fill_between(
                    [-0.02 * xMax, 0],
                    2 * [leftWidths[leftLabel]["bottom"]],
                    2 * [leftWidths[leftLabel]["bottom"] + leftWidths[leftLabel]["left"]],
                    color=colorDict[leftLabel],
                    alpha=alpha,
                    **kwargs,
                )
                ax.text(
                    -0.05 * xMax,
                    leftWidths[leftLabel]["bottom"] + 0.5 * leftWidths[leftLabel]["left"],
                    leftLabel,
                    {"ha": "right", "va": "center"},
                    fontsize=fontsize,
                )
        for rightLabel in rightLabels:
            ax.fill_between(
                [xMax + left_pos[ind], xMax + left_pos[ind]],
                2 * [rightWidths[rightLabel]["bottom"]],
                2 * [rightWidths[rightLabel]["bottom"] + rightWidths[rightLabel]["right"]],
                color=colorDict[rightLabel],
                alpha=alpha,
                **kwargs,
            )
            ax.text(
                1.05 * xMax + left_pos[ind],
                rightWidths[rightLabel]["bottom"] + 0.5 * rightWidths[rightLabel]["right"],
                rightLabel,
                {"ha": "left", "va": "center"},
                fontsize=fontsize,
            )

        if captions is not None:
            ax.text(left_pos[ind] + 0.3 * xMax, -0.1, captions[ind])

        left_pos += [left_pos[-1] + horizontal_space * xMax]

        # Plot strips
        for leftLabel in leftLabels:
            for rightLabel in rightLabels:
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

                    arr = np.linspace(0 + left_pos[ind], xMax + left_pos[ind], len(ys_d))
                    color = (
                        _color_transition(colorDict[leftLabel], colorDict[rightLabel], num=len(arr), alpha=alpha)
                        if interpolate_color
                        else colorDict[leftLabel]
                    )
                    if interpolate_color:
                        for l in range(len(ys_d)):  # necessary to get smooth lines
                            ax.fill_between(
                                arr[l:], ys_d[l:], ys_u[l:], color=color[l], ec=color[l], alpha=alpha, **kwargs
                            )
                    else:
                        ax.fill_between(arr, ys_d, ys_u, alpha=alpha, color=color, **kwargs)

        ax.axis("off")
        ax.set_title(title)
    return fig


def _heatmap(
    row_adata: AnnData,
    col_adata: AnnData,
    transition_matrix: pd.DataFrame,
    row_annotation: str,
    col_annotation: str,
    row_annotation_label: Optional[str] = None,
    col_annotation_label: Optional[str] = None,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    annotate_values: Optional[str] = "{x:.2f}",
    fontsize: float = 7.0,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[str] = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:
    cbar_kwargs = dict(cbar_kwargs)

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, dpi=dpi, figsize=figsize)
    if row_annotation != AggregationMode.CELL:
        set_palette(adata=row_adata, key=row_annotation, cont_cmap=cont_cmap)
    if col_annotation != AggregationMode.CELL:
        set_palette(adata=col_adata, key=col_annotation, cont_cmap=cont_cmap)

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

    im = ax.imshow(transition_matrix, cmap=cont_cmap, norm=norm)
    ax.grid(False)
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])

    if annotate_values is not None:
        _annotate_heatmap(transition_matrix, im, valfmt=annotate_values, cmap=cont_cmap, fontsize=fontsize, **kwargs)

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

    if col_annotation != AggregationMode.CELL:
        col_cats = divider.append_axes("top", size="2%", pad=0)
        c = fig.colorbar(col_sm, cax=col_cats, orientation="horizontal", ticklocation="top")

        c.set_ticks(np.arange(transition_matrix.shape[1]) + 0.5)
        c.ax.set_xticklabels(transition_matrix.columns, rotation=90)
        c.set_label(col_annotation if col_annotation_label is None else col_annotation_label)
    if row_annotation != AggregationMode.CELL:
        row_cats = divider.append_axes("left", size="2%", pad=0)
        c = fig.colorbar(row_sm, cax=row_cats, orientation="vertical", ticklocation="left")

        c.set_ticks(np.arange(transition_matrix.shape[0]) + 0.5)
        c.ax.set_yticklabels(transition_matrix.index[::-1])
        c.set_label(row_annotation if row_annotation_label is None else row_annotation_label)

    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def _get_black_or_white(value: float, cmap: mcolors.Colormap) -> str:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Expected value to be in interval `[0, 1]`, found `{value}`.")

    r, g, b, *_ = (int(c * 255) for c in cmap(value))
    return _contrasting_color(r, g, b)


def _annotate_heatmap(
    transition_matrix: pd.DataFrame,
    im: mpl.image.AxesImage,
    valfmt: str = "{x:.2f}",
    cmap: Union[mpl.colors.Colormap, str] = "viridis",
    fontsize: float = 5,
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
            val = im.norm(transition_matrix.iloc[i, j])
            if np.isnan(val):
                continue
            kw.update(color=_get_black_or_white(val, cmap))
            im.axes.text(j, i, valfmt(transition_matrix.iloc[i, j], None), fontsize=fontsize, **kw)


def _get_cmap_norm(
    row_adata: AnnData,
    col_adata: AnnData,
    transition_matrix: pd.DataFrame,
    row_annotation: str,
    col_annotation: str,
) -> Tuple[mcolors.ListedColormap, mcolors.ListedColormap, mcolors.BoundaryNorm, mcolors.BoundaryNorm]:

    if row_annotation != AggregationMode.CELL:
        row_color_dict = {
            row_adata.obs[row_annotation].cat.categories[i]: col
            for i, col in enumerate(row_adata.uns[f"{row_annotation}_colors"])
        }
        row_colors = [row_color_dict[cat] for cat in transition_matrix.index][::-1]
    else:
        row_colors = [0, 0, 0]
    if col_annotation != AggregationMode.CELL:
        col_color_dict = {
            col_adata.obs[col_annotation].cat.categories[i]: col
            for i, col in enumerate(col_adata.uns[f"{col_annotation}_colors"])
        }
        col_colors = [col_color_dict[cat] for cat in transition_matrix.columns]
    else:
        col_colors = [0, 0, 0]

    row_cmap = mcolors.ListedColormap(row_colors)
    col_cmap = mcolors.ListedColormap(col_colors)
    row_norm = mcolors.BoundaryNorm(np.arange(transition_matrix.shape[0] + 1), transition_matrix.shape[0])
    col_norm = mcolors.BoundaryNorm(np.arange(transition_matrix.shape[1] + 1), transition_matrix.shape[1])

    return row_cmap, col_cmap, row_norm, col_norm


def _contrasting_color(r: int, g: int, b: int) -> str:
    for val in [r, g, b]:
        assert 0 <= val <= 255, f"Color value `{val}` is not in `[0, 255]`."

    return "#000000" if r * 0.299 + g * 0.587 + b * 0.114 > 186 else "#ffffff"


def _input_to_adatas(inp: Union[AnnData, Tuple[AnnData, AnnData], CompoundProblem]) -> Tuple[AnnData, AnnData]:
    if isinstance(inp, CompoundProblem):
        return inp.adata, inp.adata
    if isinstance(inp, AnnData):
        return inp, inp
    if isinstance(inp, tuple):
        if not isinstance(inp[0], AnnData):
            raise TypeError(f"Expected object to be of type `AnnData`, found `{type(inp[0])}`.")
        if not isinstance(inp[1], AnnData):
            raise TypeError(f"Expected object to be of type `AnnData`, found `{type(inp[1])}`.")
        return inp  # type: ignore[return-value]
    raise ValueError(f"Unable to interpret input of type `{type(inp)}`.")


def _plot_temporal(
    adata: AnnData,
    temporal_key: str,
    key_stored: str,
    start: float,
    end: float,
    categories: Optional[Union[str, List[str]]] = None,
    *,
    push: bool,
    time_points: Optional[Sequence[K]] = None,
    basis: str = "umap",
    constant_fill_value: float = np.nan,
    scale: bool = True,
    cont_cmap: Union[str, mcolors.Colormap] = "viridis",
    title: Optional[Union[str, List[str]]] = None,
    suptitle: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    dot_scale_factor: float = 2.0,
    na_color: Optional[str] = "#e8ebe9",
    save: Optional[str] = None,
    ax: Optional[Axes] = None,
    show: bool = False,
    suptitle_fontsize: Optional[float] = None,
    **kwargs: Any,
) -> mpl.figure.Figure:

    fig, axs = plt.subplots(
        1, 1 if time_points is None else len(time_points), figsize=figsize, dpi=dpi, constrained_layout=True
    )
    axs = np.ravel(axs)  # make into iterable

    if not push and time_points is not None:
        time_points = time_points[::-1]

    if isinstance(title, list):
        if TYPE_CHECKING:
            assert isinstance(time_points, list)
        if len(title) != len(time_points):
            raise ValueError("If `title` is a list, its length must be equal to the length of `time_points`.")
        titles = title
    else:
        name = "descendants" if push else "ancestors"
        if time_points is not None:
            titles = [f"{categories} at time point {start if push else end}"]
            titles.extend([f"{name} at time point {time_points[i]}" for i in range(1, len(time_points))])
        else:
            titles = [f"{categories} at time point {start if push else end} and {name}"]
    for i, ax in enumerate(axs):
        if time_points is None:
            if scale:
                vmin, vmax = np.nanmin(adata.obs[key_stored]), np.nanmax(adata.obs[key_stored])
            else:
                vmin, vmax = 0, 1
            adata.obs[f"result_key_{i}"] = (adata.obs[key_stored] - vmin) / (vmax - vmin)
            size = None
        else:
            tmp = np.full(len(adata), constant_fill_value)
            mask = adata.obs[temporal_key] == time_points[i]

            tmp[mask] = adata[adata.obs[temporal_key] == time_points[i]].obs[key_stored]
            if scale:
                vmin = np.nanmin(adata.obs[key_stored] * mask)
                vmax = np.nanmax(adata.obs[key_stored] * mask)
            else:
                vmin = 0
                vmax = 1
            adata.obs[f"result_key_{i}"] = (tmp - vmin) / (vmax - vmin)
            size = (mask * 120000 * dot_scale_factor + (1 - mask) * 120000) / adata.n_obs  # 120,000 from scanpy

            _ = kwargs.pop("color_map", None)
            _ = kwargs.pop("palette", None)

            if isinstance(cont_cmap, str):
                try:
                    cont_cmap = mpl.colormaps[cont_cmap]
                except KeyError:
                    raise KeyError("`cont_cmap` is not a valid key for `matplotlib.colormaps`.")
            if (time_points[i] == start and push) or (time_points[i] == end and not push):
                vmin, vmax = np.nanmin(adata.obs[f"result_key_{i}"]), np.nanmax(adata.obs[f"result_key_{i}"])
                adata.obs[f"result_key_{i}"] = adata.obs[f"result_key_{i}"].astype("str")
                st = f"not in {time_points[i]}"
                adata.obs[f"result_key_{i}"] = adata.obs[f"result_key_{i}"].replace("nan", st)
                palette = {str(vmax): cont_cmap.reversed()(0), str(vmin): cont_cmap(0), st: na_color}
                kwargs["palette"] = palette
            else:
                kwargs["color_map"] = cont_cmap

        sc.pl.embedding(
            adata=adata,
            basis=basis,
            color=f"result_key_{i}",
            title=titles[i],
            size=size,
            ax=ax,
            show=show,
            na_color=na_color,
            **kwargs,
        )
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    if save:
        fig.figure.savefig(save, bbox_inches="tight")
    return fig


def _color_transition(c1: str, c2: str, num: int, alpha: float) -> List[str]:
    if not mpl.colors.is_color_like(c1):
        raise ValueError(f"{c1} cannot be interpreted as an RGB color.")
    if not mpl.colors.is_color_like(c2):
        raise ValueError(f"{c2} cannot be interpreted as an RGB color.")
    c1_rgb = np.array(mpl.colors.to_rgb(c1))
    c2_rgb = np.array(mpl.colors.to_rgb(c2))
    return [mpl.colors.to_rgb((1 - n / num) * c1_rgb + n / num * c2_rgb) + (alpha,) for n in range(num)]


def _create_col_colors(adata: AnnData, obs_col: str, subset: Union[str, Sequence[str]]) -> Optional[mcolors.Colormap]:

    if isinstance(subset, Sequence) and len(subset) == 1 and isinstance(subset[0], str):
        subset = subset[0]
    if not is_categorical_dtype(adata.obs[obs_col]):
        raise TypeError(f"`adata.obs['{obs_col}'] must be of categorical type.")
    if isinstance(subset, str):
        try:
            colorDict = {
                cat: adata.uns[f"{obs_col}_colors"][i] for i, cat in enumerate(adata.obs[obs_col].cat.categories)
            }
        except KeyError:
            raise KeyError(f"Unable to access `adata.uns['{obs_col}_colors']`.") from None

        color = colorDict[subset]

        h, _, v = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
        end_color = mcolors.hsv_to_rgb([h, 1, v])

        col_cmap = mcolors.LinearSegmentedColormap.from_list("category_cmap", ["darkgrey", end_color])

        return col_cmap

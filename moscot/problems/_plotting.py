from typing import Any, Dict, List, Union, Optional
from pathlib import Path
from collections import defaultdict
import os

from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from scanpy import logging as logg, settings
from anndata import AnnData
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation as add_color_palette

from moscot._docs import d


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
    palette: Union[str, ListedColormap] = "viridis",
    force_update_colors: bool = False,
    **_: Any,
) -> None:
    if key not in adata.obs.columns:
        raise KeyError("TODO: invalid key.")
    uns_key = f"{key}_colors"
    if uns_key not in adata.uns:
        add_color_palette(adata, key=key, palette=palette, force_update_colors=force_update_colors)


def _sankey(
    adata: AnnData,
    key: str,
    transition_matrices: List[pd.DataFrame],
    captions: Optional[List[str]] = None,
    colorDict: Optional[Union[Dict[Any, str], ListedColormap]] = None,
    title: Optional[str] = None,
    **kwargs: Any,
) -> None:
    if captions is not None and len(captions) != len(transition_matrices):
        raise ValueError("TODO: If `captions` are specified length has to be same as of `transition_matrices`.")
    if colorDict is None:
        # TODO: adapt for unique categories
        set_palette(adata=adata, key=key, **kwargs)

        colorDict = {cat: adata.uns[f"{key}_colors"][i] for i, cat in enumerate(adata.obs[key].cat.categories)}
    else:
        missing = [label for label in adata.obs[key].cat.categories if label not in colorDict.keys()]
        if missing:
            msg = "The colorDict parameter is missing values for the following labels : "
            msg += "{}".format(", ".join(missing))
            raise ValueError(msg)
    fontsize = kwargs.pop("fontsize", 12)
    horizontal_space = kwargs.pop("horizontal_space", 1.5)
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

from abc import ABC, ABCMeta
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import pytest
from anndata import AnnData

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from moscot import _constants
from moscot.plotting._utils import set_plotting_vars

HERE: Path = Path(__file__).parent

EXPECTED = HERE / "expected_figures"
ACTUAL = HERE / "actual_figures"
TOL = 60
DPI = 40


@pytest.fixture()
def adata_pl_cell_transition(gt_temporal_adata: AnnData) -> AnnData:
    plot_vars = {
        "transition_matrix": gt_temporal_adata.uns["cell_transition_10_105_forward"],
        "source_groups": "cell_type",
        "target_groups": "cell_type",
        "source": 0,
        "target": 1,
    }
    set_plotting_vars(gt_temporal_adata, _constants.CELL_TRANSITION, key=_constants.CELL_TRANSITION, value=plot_vars)

    return gt_temporal_adata


@pytest.fixture()
def adata_pl_push(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    plot_vars = {"temporal_key": "time", "data": "celltype", "subset": "A", "source": 0, "target": 1}
    adata_time.uns["celltype_colors"] = ["#cc1b1b", "#2ccc1b", "#cc1bcc"]
    adata_time.obs["celltype"] = adata_time.obs["celltype"].astype("category")
    set_plotting_vars(adata_time, _constants.PUSH, key=_constants.PUSH, value=plot_vars)
    push_initial_dist = np.zeros(
        shape=(len(adata_time[adata_time.obs["time"] == 0]),)
    )  # we need this for a cat. distr. in plots
    push_initial_dist[0:10] = 0.1
    nan2 = np.empty(len(adata_time[adata_time.obs["time"] == 2]))
    nan2[:] = np.nan
    adata_time.obs[_constants.PUSH] = np.hstack(
        (push_initial_dist, np.abs(rng.randn(len(adata_time[adata_time.obs["time"] == 1]))), nan2)
    )
    return adata_time


@pytest.fixture()
def adata_pl_pull(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    plot_vars = {"temporal_key": "time", "data": "celltype", "subset": "A", "source": 0, "target": 1}
    adata_time.uns["celltype_colors"] = ["#cc1b1b", "#2ccc1b", "#cc1bcc"]
    adata_time.obs["celltype"] = adata_time.obs["celltype"].astype("category")
    set_plotting_vars(adata_time, _constants.PULL, key=_constants.PULL, value=plot_vars)
    pull_initial_dist = np.zeros(
        shape=(len(adata_time[adata_time.obs["time"] == 1]),)
    )  # we need this for a cat. distr. in plots
    pull_initial_dist[0:10] = 0.1
    rand0 = np.abs(rng.randn(len(adata_time[adata_time.obs["time"] == 0])))
    nan2 = np.empty(len(adata_time[adata_time.obs["time"] == 2]))
    nan2[:] = np.nan
    adata_time.obs[_constants.PULL] = np.hstack((rand0, pull_initial_dist, nan2))
    return adata_time


@pytest.fixture()
def adata_pl_sankey(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    celltypes = ["A", "B", "C", "D", "E"]
    adata_time.obs["celltype"] = rng.choice(celltypes, size=len(adata_time))
    adata_time.obs["celltype"] = adata_time.obs["celltype"].astype("category")
    data1 = np.abs(rng.randn(5, 5))
    data2 = np.abs(rng.randn(5, 5))
    tm1 = pd.DataFrame(data=data1, index=celltypes, columns=celltypes)
    tm2 = pd.DataFrame(data=data2, index=celltypes, columns=celltypes)
    plot_vars = {"transition_matrices": [tm1, tm2], "captions": ["0", "1"], "key": "celltype"}
    set_plotting_vars(adata_time, _constants.SANKEY, key=_constants.SANKEY, value=plot_vars)

    return adata_time


def _decorate(fn: Callable, clsname: str, name: Optional[str] = None) -> Callable:
    @wraps(fn)
    def save_and_compare(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.compare(fig_name)

    if not callable(fn):
        raise TypeError(f"Expected a `callable` for class `{clsname}`, found `{type(fn).__name__}`.")

    name = fn.__name__ if name is None else name

    if not name.startswith("test_plot_") or not clsname.startswith("Test"):
        return fn

    fig_name = f"{clsname[4:]}_{name[10:]}"

    return save_and_compare


class PlotTesterMeta(ABCMeta):
    def __new__(cls, clsname, superclasses, attributedict):
        for key, value in attributedict.items():
            if callable(value):
                attributedict[key] = _decorate(value, clsname, name=key)
        return super().__new__(cls, clsname, superclasses, attributedict)


# ideally, we would you metaclass=PlotTesterMeta and all plotting tests just subclass this
# but for some reason, pytest erases the metaclass info
class PlotTester(ABC):  # noqa: B024
    @classmethod
    def compare(cls, basename: str, tolerance: Optional[float] = None):
        ACTUAL.mkdir(parents=True, exist_ok=True)
        out_path = ACTUAL / f"{basename}.png"

        plt.savefig(out_path, dpi=DPI)
        plt.close()

        tolerance = TOL if tolerance is None else tolerance

        res = compare_images(str(EXPECTED / f"{basename}.png"), str(out_path), tolerance)

        assert res is None, res

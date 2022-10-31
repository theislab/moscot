from abc import ABC, ABCMeta
from typing import Callable, Optional
from pathlib import Path
from functools import wraps

from matplotlib.testing.compare import compare_images
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import numpy as np

from anndata import AnnData

from moscot._constants._constants import Key, AdataKeys, PlottingKeys, PlottingDefaults

HERE: Path = Path(__file__).parent

EXPECTED = HERE / "_images"
ACTUAL = HERE / "figures"
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
    Key.uns.set_plotting_vars(
        gt_temporal_adata, AdataKeys.UNS, PlottingKeys.CELL_TRANSITION, PlottingDefaults.CELL_TRANSITION, plot_vars
    )

    return gt_temporal_adata


@pytest.fixture()
def adata_pl_push(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    plot_vars = {"temporal_key": "time"}
    Key.uns.set_plotting_vars(adata_time, AdataKeys.UNS, PlottingKeys.PUSH, PlottingDefaults.PUSH, plot_vars)
    adata_time.obs[PlottingDefaults.PUSH] = np.abs(rng.randn(len(adata_time)))
    return adata_time


@pytest.fixture()
def adata_pl_pull(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(0)
    plot_vars = {"temporal_key": "time"}
    Key.uns.set_plotting_vars(adata_time, AdataKeys.UNS, PlottingKeys.PULL, PlottingDefaults.PULL, plot_vars)
    adata_time.obs[PlottingDefaults.PULL] = np.abs(rng.randn(len(adata_time)))
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
    Key.uns.set_plotting_vars(adata_time, AdataKeys.UNS, PlottingKeys.SANKEY, PlottingDefaults.SANKEY, plot_vars)

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

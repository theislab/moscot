from anndata import AnnData

import moscot.plotting as mpl
from tests.plotting.conftest import PlotTester, PlotTesterMeta

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be changed, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestPlotting(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_cell_transition(self, adata_pl_cell_transition: AnnData):
        mpl.cell_transition(adata_pl_cell_transition)

    def test_plot_cell_transition_params(self, adata_pl_cell_transition: AnnData):
        mpl.cell_transition(adata_pl_cell_transition, annotate=None, cmap="inferno", fontsize=15)

    def test_plot_sankey(self, adata_pl_sankey: AnnData):
        mpl.sankey(adata_pl_sankey)

    def test_plot_sankey_params(self, adata_pl_sankey: AnnData):
        mpl.sankey(adata_pl_sankey, captions=["Test", "Other test"], title="Title")

    def test_plot_push(self, adata_pl_push: AnnData):
        mpl.push(adata_pl_push, time_points=[0, 1])

    def test_plot_pull(self, adata_pl_pull: AnnData):
        mpl.pull(adata_pl_pull, time_points=[0, 1])

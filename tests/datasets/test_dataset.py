from typing import Mapping, Optional

import pytest

import networkx as nx
import numpy as np

from moscot.datasets import simulate_data


class TestSimulateData:
    @pytest.mark.fast
    @pytest.mark.parametrize("n_distributions", [2, 4])
    @pytest.mark.parametrize("key", ["batch", "day"])
    def test_n_distributions(self, n_distributions: int, key: str):
        adata = simulate_data(n_distributions=n_distributions, key=key)
        assert key in adata.obs.columns
        assert adata.obs[key].nunique() == n_distributions

    @pytest.mark.fast
    @pytest.mark.parametrize("obs_to_add", [{"celltype": 2}, {"celltype": 5, "cluster": 4}])
    def test_obs_to_add(self, obs_to_add: Mapping[str, int]):
        adata = simulate_data(obs_to_add=obs_to_add)

        for colname, k in obs_to_add.items():
            assert colname in adata.obs.columns
            assert adata.obs[colname].nunique() == k

    @pytest.mark.fast
    @pytest.mark.parametrize("spatial_dim", [None, 2, 3])
    def test_quad_term_spatial(self, spatial_dim: Optional[int]):
        kwargs = {}
        if spatial_dim is not None:
            kwargs["spatial_dim"] = spatial_dim
        adata = simulate_data(quad_term="spatial", **kwargs)

        assert isinstance(adata.obsm["spatial"], np.ndarray)
        if spatial_dim is None:
            assert adata.obsm["spatial"].shape[1] == 2
        else:
            assert adata.obsm["spatial"].shape[1] == spatial_dim

    @pytest.mark.fast
    @pytest.mark.parametrize("n_intBCs", [None, 4, 7])
    @pytest.mark.parametrize("barcode_dim", [None, 5, 8])
    def test_quad_term_barcode(self, n_intBCs: Optional[int], barcode_dim: Optional[int]):
        kwargs = {}
        if n_intBCs is not None:
            kwargs["n_intBCs"] = n_intBCs
        if barcode_dim is not None:
            kwargs["barcode_dim"] = barcode_dim

        adata = simulate_data(quad_term="barcode", **kwargs)

        assert isinstance(adata.obsm["barcode"], np.ndarray)
        if barcode_dim is None:
            assert adata.obsm["barcode"].shape[1] == 10
        else:
            assert adata.obsm["barcode"].shape[1] == barcode_dim

        if n_intBCs is None:
            assert len(np.unique(adata.obsm["barcode"])) <= 20
        else:
            assert len(np.unique(adata.obsm["barcode"])) <= n_intBCs

    @pytest.mark.fast
    @pytest.mark.parametrize("n_initial_nodes", [None, 4, 7])
    @pytest.mark.parametrize("n_distributions", [3, 6])
    def test_quad_term_tree(self, n_initial_nodes: Optional[int], n_distributions: int):
        kwargs = {}
        if n_initial_nodes is not None:
            kwargs["n_initial_nodes"] = n_initial_nodes
        adata = simulate_data(quad_term="tree", key="day", n_distributions=n_distributions, **kwargs)

        assert isinstance(adata.uns["trees"], dict)
        assert len(adata.uns["trees"]) == n_distributions

        for i in range(len(adata.uns["trees"])):
            assert isinstance(adata.uns["trees"][i], nx.DiGraph)

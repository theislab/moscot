import pytest

import numpy as np
import pandas as pd

import anndata as ad

from moscot.costs._costs import BarcodeDistance, _scaled_hamming_dist
from moscot.costs._utils import get_cost


class TestBarcodeDistance:
    RNG = np.random.RandomState(0)

    @staticmethod
    def test_barcode_distance_init():
        adata = ad.AnnData(TestBarcodeDistance.RNG.rand(3, 3), obsm={"barcodes": TestBarcodeDistance.RNG.rand(3, 3)})
        # initialization failure when no adata is provided
        with pytest.raises(TypeError):
            get_cost("barcode_distance", backend="moscot")
        # initialization failure when invalid key is provided
        with pytest.raises(KeyError):
            get_cost("barcode_distance", backend="moscot", adata=adata, key="invalid_key", attr="obsm")
        # initialization failure when invalid attr
        with pytest.raises(AttributeError):
            get_cost("barcode_distance", backend="moscot", adata=adata, key="barcodes", attr="invalid_attr")
        # check if not None
        cost_fn: BarcodeDistance = get_cost(
            "barcode_distance", backend="moscot", adata=adata, key="barcodes", attr="obsm"
        )
        assert cost_fn is not None

    @staticmethod
    def test_scaled_hamming_dist_with_sample_inputs():
        # Sample input arrays
        x = np.array([1, -1, 0, 1])
        y = np.array([0, 1, 1, 1])

        # Expected output
        expected_distance = 2.0 / 3

        # Compute the scaled Hamming distance
        computed_distance = _scaled_hamming_dist(x, y)

        # Check if the computed distance matches the expected distance
        np.testing.assert_almost_equal(computed_distance, expected_distance, decimal=4)

    @staticmethod
    def test_scaled_hamming_dist_if_nan():
        # Sample input arrays with no shared indices
        x = np.array([-1, -1, 0, 1])
        y = np.array([0, 1, -1, -1])

        with pytest.raises(ValueError, match="No shared indices."):
            _scaled_hamming_dist(x, y)

    @staticmethod
    def test_barcode_distance_with_sample_input():
        # Example barcodes
        barcodes = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])

        # Create a dummy AnnData object with the example barcodes
        adata = ad.AnnData(TestBarcodeDistance.RNG.rand(3, 3))
        adata.obsm["barcodes"] = barcodes

        # Initialize BarcodeDistance
        cost_fn: BarcodeDistance = get_cost(
            "barcode_distance", backend="moscot", adata=adata, key="barcodes", attr="obsm"
        )

        # Compute distances
        computed_distances = cost_fn()

        # Expected distance matrix
        expected_distances = np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]]) / 3.0

        # Check if the computed distances match the expected distances
        np.testing.assert_almost_equal(computed_distances, expected_distances, decimal=4)


class TestLeafDistance:
    @staticmethod
    def create_dummy_adata_leaf():
        import networkx as nx

        adata: ad.AnnData = ad.AnnData(
            X=np.ones((10, 10)),
            obs=pd.DataFrame(data={"day": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]}),
        )
        g: nx.DiGraph = nx.DiGraph()
        g.add_nodes_from([str(i) for i in range(3)] + ["root"])
        g.add_edges_from([("root", str(i)) for i in range(3)])
        adata.uns["tree"] = {0: g}
        return adata

    @staticmethod
    def test_leaf_distance_init():
        adata = TestLeafDistance.create_dummy_adata_leaf()
        # initialization failure when no adata is provided
        with pytest.raises(TypeError):
            get_cost("leaf_distance", backend="moscot")
        # initialization failure when invalid key is provided
        with pytest.raises(KeyError, match="Unable to find tree in"):
            get_cost("leaf_distance", backend="moscot", adata=adata, key="invalid_key", attr="uns", dist_key=0)
        # initialization failure when invalid dist_key is provided
        with pytest.raises(KeyError, match="Unable to find tree in"):
            get_cost("leaf_distance", backend="moscot", adata=adata, key="tree", attr="uns")
        # when leaves do not match adata.obs_names
        with pytest.raises(ValueError, match="Leaves do not match"):
            get_cost("leaf_distance", backend="moscot", adata=adata, key="tree", attr="uns", dist_key=0)()
        # now giving valid input
        adata0 = adata[adata.obs.day == 0]
        cost_fn = get_cost("leaf_distance", backend="moscot", adata=adata0, key="tree", attr="uns", dist_key=0)
        np.testing.assert_equal(cost_fn(), np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]]))
        # when tree is not a networkx.Graph
        adata0.uns["tree"] = {0: 1}
        with pytest.raises(TypeError, match="networkx.Graph"):
            get_cost("leaf_distance", backend="moscot", adata=adata0, key="tree", attr="uns", dist_key=0)

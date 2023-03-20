from typing import Tuple

import pytest

import numpy as np
from scipy.sparse import csr_matrix

from tests._utils import ATOL, RTOL, MockSolverOutput


class TestBaseSolverOutput:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("threshold", [0.0, 1e-1, 1.0])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_threshold(self, batch_size: int, threshold: float, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = np.abs(rng.rand(shape[0], shape[1]))
        output = MockSolverOutput(tmap)
        output.sparsify(mode="threshold", threshold=threshold, batch_size=batch_size)
        res = output.sparse_transport_matrix
        assert isinstance(res, csr_matrix)
        assert res.shape == shape
        assert np.all(res.data >= 0)
        if threshold == 0.0:
            np.testing.assert_allclose(res.A, tmap, rtol=RTOL, atol=ATOL)
        elif threshold == 1e-1:
            data = res.data
            assert np.sum((data > threshold) + (data == 0)) == len(data)
        elif threshold == 1.0:
            assert res.nnz == 0
        else:
            raise ValueError(f"Threshold {threshold} not expected.")

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_minrow(self, batch_size: int, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = np.abs(rng.rand(shape[0], shape[1])) + 1e-3  # make sure it's not 0
        output = MockSolverOutput(tmap)
        output.sparsify(mode="min_row", batch_size=batch_size)
        res = output.sparse_transport_matrix
        assert isinstance(res, csr_matrix)
        assert res.shape == shape
        assert np.all(res.data >= 0.0)
        assert np.all(np.sum(res.A, axis=1) > 0.0)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("threshold", [0, 10, 100])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_percentile(self, batch_size: int, threshold: float, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = np.abs(rng.rand(shape[0], shape[1])) + 1e-3  # make sure it's not 0
        output = MockSolverOutput(tmap)
        output.sparsify(mode="percentile", threshold=threshold, batch_size=batch_size, n_samples=shape[0], seed=42)
        res = output.sparse_transport_matrix
        assert isinstance(res, csr_matrix)
        assert res.shape == shape
        assert np.all(res.data >= 0.0)
        if threshold == 0:
            assert np.sum(tmap != res.A) < res.shape[0] * res.shape[1] * 0.1  # this only holds with probability < 1
        if threshold == 100:
            assert res.nnz < res.shape[0] * res.shape[1] * 0.9  # this only holds with probability < 1

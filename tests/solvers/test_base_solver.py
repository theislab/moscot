from typing import Tuple

import pytest

import numpy as np
import scipy.sparse as sp

from moscot.base.output import MatrixSolverOutput
from tests._utils import ATOL, RTOL, MockSolverOutput


class TestBaseSolverOutput:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("threshold", [0.0, 1e-1, 1.0])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_threshold(self, batch_size: int, threshold: float, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = np.abs(rng.rand(shape[0], shape[1]))
        output = MockSolverOutput(tmap)
        mso = output.sparsify(mode="threshold", value=threshold, batch_size=batch_size)
        assert isinstance(mso, MatrixSolverOutput)
        res = mso.transport_matrix
        assert isinstance(res, sp.csr_matrix)
        assert res.shape == shape
        np.testing.assert_array_equal(res.data >= 0.0, np.full(res.data.shape, True))
        if threshold == 0.0:
            np.testing.assert_allclose(res.A, tmap, rtol=RTOL, atol=ATOL)
        elif threshold == 1e-1:
            data = res.data
            np.testing.assert_equal(np.sum((data > threshold) + (data == 0)), len(data))
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
        mso = output.sparsify(mode="min_row", batch_size=batch_size)
        assert isinstance(mso, MatrixSolverOutput)
        res = mso.transport_matrix
        assert isinstance(res, sp.csr_matrix)
        assert res.shape == shape
        np.testing.assert_array_equal(res.data >= 0.0, np.full(res.data.shape, True))
        np.testing.assert_array_equal(np.sum(res.A, axis=1) > 0.0, np.full((shape[0],), True))

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("threshold", [0, 10, 100])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_percentile(self, batch_size: int, threshold: float, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = np.abs(rng.rand(shape[0], shape[1])) + 1e-3  # make sure it's not 0
        output = MockSolverOutput(tmap)
        mso = output.sparsify(mode="percentile", value=threshold, batch_size=batch_size, n_samples=shape[0], seed=42)
        assert isinstance(mso, MatrixSolverOutput)
        res = mso.transport_matrix
        assert isinstance(res, sp.csr_matrix)
        assert res.shape == shape
        np.testing.assert_array_equal(res.data >= 0.0, np.full(res.data.shape, True))
        n, m = shape
        if threshold == 0:
            assert np.sum(tmap != res.A) < n * m * 0.1  # this only holds with probability < 1
        if threshold == 100:
            assert res.nnz < n * m * 0.9  # this only holds with probability < 1

from typing import Tuple

import pytest

import jax.numpy as jnp
import jax.random
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
        np.testing.assert_array_equal(res.data >= 0.0, True)
        vec_pull = np.abs(rng.randn(shape[1], 1))
        pull1 = mso.pull(vec_pull)
        pull2 = output.pull(vec_pull)
        assert isinstance(pull1, np.ndarray)

        if threshold == 0.0:
            np.testing.assert_allclose(res.A, tmap, rtol=RTOL, atol=ATOL)
            np.testing.assert_array_less(0.5, np.corrcoef(pull1.squeeze(), pull2.squeeze())[0, 1])
        elif threshold == 1e-1:
            data = res.data
            np.testing.assert_equal(np.sum((data > threshold) + (data == 0)), len(data))
            np.testing.assert_array_less(0.5, np.corrcoef(pull1.squeeze(), pull2.squeeze())[0, 1])
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
        np.testing.assert_array_equal(res.data >= 0.0, True)
        np.testing.assert_array_equal(np.sum(res.A, axis=1) > 0.0, True)
        vec_pull = np.abs(rng.randn(shape[1], 1))
        pull1 = mso.pull(vec_pull)
        pull2 = output.pull(vec_pull)
        assert isinstance(pull1, np.ndarray)
        np.testing.assert_array_less(0.5, np.corrcoef(pull1.squeeze(), pull2.squeeze())[0, 1])

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("threshold", [0, 10, 100])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_percentile(self, batch_size: int, threshold: float, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), shape=shape)) + 1e-3
        output = MockSolverOutput(tmap)
        mso = output.sparsify(mode="percentile", value=threshold, batch_size=batch_size, n_samples=shape[0], seed=42)
        assert isinstance(mso, MatrixSolverOutput)
        res = mso.transport_matrix
        assert isinstance(res, sp.csr_matrix)
        assert res.shape == shape
        np.testing.assert_array_equal(res.data >= 0.0, True)
        n, m = shape
        if threshold == 0:
            assert np.sum(tmap != res.A) < n * m * 0.1  # this only holds with probability < 1
        if threshold == 100:
            assert res.nnz < n * m * 0.9  # this only holds with probability < 1
        vec_pull = np.abs(rng.randn(shape[1], 1))
        pull1 = mso.pull(vec_pull)
        pull2 = output.pull(vec_pull)
        assert isinstance(pull1, np.ndarray)
        if threshold < 100:
            np.testing.assert_array_less(0.5, np.corrcoef(pull1.squeeze(), pull2.squeeze())[0, 1])

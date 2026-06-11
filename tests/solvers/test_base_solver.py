from typing import Tuple

import pytest

import jax.numpy as jnp
import jax.random
import numpy as np
import scipy.sparse as sp

from moscot.base.output import MatrixSolverOutput
from tests._utils import ATOL, RTOL, MockSolverOutput


class TestBaseDiscreteSolverOutput:
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
            np.testing.assert_allclose(res.toarray(), tmap, rtol=RTOL, atol=ATOL)
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
        np.testing.assert_array_equal(np.sum(res.toarray(), axis=1) > 0.0, True)
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
            assert np.sum(tmap != res.toarray()) < n * m * 0.1  # this only holds with probability < 1
        if threshold == 100:
            assert res.nnz < n * m * 0.9  # this only holds with probability < 1
        vec_pull = np.abs(rng.randn(shape[1], 1))
        pull1 = mso.pull(vec_pull)
        pull2 = output.pull(vec_pull)
        assert isinstance(pull1, np.ndarray)
        if threshold < 100:
            np.testing.assert_array_less(0.5, np.corrcoef(pull1.squeeze(), pull2.squeeze())[0, 1])

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103), (103, 91)])
    def test_sparsify_mass_reconstruction(self, batch_size: int, shape: Tuple[int, int]) -> None:
        # `value=1.0, max_k=None` must reproduce the full transport matrix (exact-reconstruction oracle).
        rng = np.random.RandomState(0)
        tmap = np.abs(rng.rand(shape[0], shape[1]))
        output = MockSolverOutput(tmap)
        mso = output.sparsify(mode="mass", value=1.0, max_k=None, batch_size=batch_size)
        assert isinstance(mso, MatrixSolverOutput)
        res = mso.transport_matrix
        assert isinstance(res, sp.csr_matrix)
        assert res.shape == shape
        np.testing.assert_allclose(res.toarray(), tmap, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("value", [0.5, 0.9])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103), (103, 91)])
    def test_sparsify_mass_retention(self, value: float, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(0)
        tmap = np.abs(rng.rand(shape[0], shape[1]))
        output = MockSolverOutput(tmap)
        res = output.sparsify(mode="mass", value=value, batch_size=4).transport_matrix.toarray()
        # each row retains at least `value` of its original mass, and the result is a subset of `tmap`.
        np.testing.assert_array_less(value * tmap.sum(axis=1) - 1e-9, res.sum(axis=1) + 1e-9)
        kept = res > 0.0
        np.testing.assert_allclose(res[kept], tmap[kept], rtol=RTOL, atol=ATOL)

    def test_sparsify_mass_max_k(self) -> None:
        rng = np.random.RandomState(0)
        tmap = np.abs(rng.rand(20, 30))
        output = MockSolverOutput(tmap)
        # `max_k` caps the number of entries per row.
        res = output.sparsify(mode="mass", value=1.0, max_k=3, batch_size=4).transport_matrix
        assert np.all(np.diff(res.indptr) <= 3)
        # with `max_k=1` only the per-row argmax survives.
        res1 = output.sparsify(mode="mass", value=0.99, max_k=1, batch_size=4).transport_matrix
        assert np.all(np.diff(res1.indptr) <= 1)
        np.testing.assert_array_equal(res1.toarray().argmax(axis=1), tmap.argmax(axis=1))

    def test_sparsify_mass_empty_rows(self) -> None:
        rng = np.random.RandomState(0)
        tmap = np.abs(rng.rand(6, 5))
        tmap[2] = 0.0  # all-zero row must keep nothing
        output = MockSolverOutput(tmap)
        res = output.sparsify(mode="mass", value=0.9, batch_size=2).transport_matrix
        assert res[2].nnz == 0

    def test_sparsify_mass_validation(self) -> None:
        output = MockSolverOutput(np.abs(np.random.RandomState(0).rand(5, 4)))
        with pytest.raises(ValueError, match="value"):
            output.sparsify(mode="mass")
        with pytest.raises(ValueError, match="value"):
            output.sparsify(mode="mass", value=1.5)
        with pytest.raises(ValueError, match="value"):
            output.sparsify(mode="mass", value=0.0)
        with pytest.raises(ValueError, match="max_k"):
            output.sparsify(mode="mass", value=0.5, max_k=0)
        with pytest.raises(ValueError, match="max_k"):
            output.sparsify(mode="threshold", value=0.5, max_k=3)

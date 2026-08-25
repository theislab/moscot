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
        # `min_row` uses the largest threshold that keeps every row, `min_i max_j T_ij`; retained
        # entries are unchanged. (A `pull` correlation proxy is not meaningful here: on a uniform
        # random matrix every row maximum is alike, so the rule keeps ~1 entry per row by design.
        # `TestMinRowStructure` checks the proxy on a peaked, transport-plan-like matrix instead.)
        np.testing.assert_allclose(
            res.toarray(), np.where(tmap >= tmap.max(axis=1).min(), tmap, 0.0), rtol=RTOL, atol=ATOL
        )
        vec_pull = np.abs(rng.randn(shape[1], 1))
        pull1 = mso.pull(vec_pull)
        assert isinstance(pull1, np.ndarray)

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

    @pytest.mark.parametrize("batch_size", [1, 4, 103, 1000])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_minrow_batch_independent(self, batch_size: int, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = np.abs(rng.rand(shape[0], shape[1])) + 1e-3
        res = MockSolverOutput(tmap).sparsify(mode="min_row", batch_size=batch_size).transport_matrix
        # exact rule: the largest threshold that keeps every row, `min_i max_j T_ij`
        expected = np.where(tmap >= tmap.max(axis=1).min(), tmap, 0.0)
        np.testing.assert_allclose(res.toarray(), expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("batch_size", [1, 4, 91, 1000])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_minrow_every_row_nonempty(self, batch_size: int, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = np.abs(rng.rand(shape[0], shape[1])) + 1e-3
        res = MockSolverOutput(tmap).sparsify(mode="min_row", batch_size=batch_size).transport_matrix
        assert np.all(np.diff(res.indptr) >= 1)

    @pytest.mark.parametrize("value", [0.0, 50.0, 100.0])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify_percentile_samples_rows(self, value: float, shape: Tuple[int, int]) -> None:
        # with `n_samples = n` every row is sampled, so the threshold is the exact matrix percentile
        rng = np.random.RandomState(0)
        tmap = np.abs(rng.rand(shape[0], shape[1])) + 1e-3
        mso = MockSolverOutput(tmap).sparsify(mode="percentile", value=value, n_samples=shape[0], seed=42)
        expected = np.where(tmap >= np.percentile(tmap, value), tmap, 0.0)
        np.testing.assert_allclose(mso.transport_matrix.toarray(), expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize(
        ("mode", "kwargs"),
        [
            ("min_row", {}),
            ("threshold", {"value": 0.1}),
            ("percentile", {"value": 50.0}),
            ("mass", {"value": 0.9}),
        ],
    )
    def test_sparsify_does_not_apply_transport(self, mode: str, kwargs: dict, monkeypatch) -> None:
        rng = np.random.RandomState(0)
        output = MockSolverOutput(np.abs(rng.rand(9, 5)) + 1e-3)

        def _boom(*_args, **_kwargs):
            raise AssertionError("`sparsify` must not apply the transport matrix")

        monkeypatch.setattr(MockSolverOutput, "push", _boom, raising=False)
        monkeypatch.setattr(MockSolverOutput, "pull", _boom, raising=False)
        output.sparsify(mode=mode, batch_size=2, **kwargs)

    def test_sparsify_propagates_metadata(self) -> None:
        rng = np.random.RandomState(0)
        output = MockSolverOutput(np.abs(rng.rand(6, 4)) + 1e-3)
        mso = output.sparsify(mode="min_row", batch_size=2)
        assert mso.cost == output.cost
        assert mso.converged == output.converged
        assert mso.is_linear == output.is_linear


class TestMinRowStructure:
    """`min_row` on a transport-plan-like matrix: sparse, yet still faithful under `pull`."""

    @staticmethod
    def _peaked(shape: Tuple[int, int], seed: int = 0) -> np.ndarray:
        # rows concentrated around a moving optimum, as an entropic plan is
        rng = np.random.RandomState(seed)
        n, m = shape
        centers = np.linspace(0, m - 1, n)[:, None]
        tmap = np.exp(-((np.arange(m)[None, :] - centers) ** 2) / (2.0 * (m / 20.0) ** 2))
        tmap *= 1.0 + 0.1 * rng.rand(n, m)
        return tmap / tmap.sum(axis=1, keepdims=True)

    @pytest.mark.parametrize("batch_size", [1, 4, 128])
    def test_minrow_preserves_pull(self, batch_size: int) -> None:
        tmap = self._peaked((91, 103))
        output = MockSolverOutput(tmap)
        mso = output.sparsify(mode="min_row", batch_size=batch_size)
        vec_pull = np.abs(np.random.RandomState(1).randn(tmap.shape[1], 1))
        pull1, pull2 = mso.pull(vec_pull), output.pull(vec_pull)
        np.testing.assert_array_less(0.5, np.corrcoef(pull1.squeeze(), pull2.squeeze())[0, 1])

    def test_minrow_is_sparse_and_keeps_every_row_argmax(self) -> None:
        tmap = self._peaked((91, 103))
        res = MockSolverOutput(tmap).sparsify(mode="min_row", batch_size=8).transport_matrix
        assert res.nnz < 0.1 * tmap.size  # a sparsification mode must actually sparsify
        assert np.all(np.diff(res.indptr) >= 1)
        # `min_row` guarantees each row keeps at least its largest entry (mass retention is `mode='mass'`)
        dense = res.toarray()
        rows = np.arange(tmap.shape[0])
        np.testing.assert_allclose(dense[rows, tmap.argmax(axis=1)], tmap.max(axis=1), rtol=RTOL, atol=ATOL)


class TestMasslessRows:
    """Rows without mass are handled the same way by every mode: kept empty, ignored by thresholds."""

    @staticmethod
    def _tmap(seed: int = 0) -> np.ndarray:
        tmap = np.abs(np.random.RandomState(seed).rand(8, 6)) + 1e-3
        tmap[3] = 0.0  # e.g. a strongly unbalanced solution
        return tmap

    @pytest.mark.parametrize(
        ("mode", "kwargs"),
        [
            ("min_row", {}),
            ("threshold", {"value": 0.5}),
            ("percentile", {"value": 50.0, "n_samples": 8, "seed": 0}),
            ("mass", {"value": 0.9}),
        ],
    )
    def test_massless_row_stays_empty(self, mode: str, kwargs: dict) -> None:
        res = MockSolverOutput(self._tmap()).sparsify(mode=mode, batch_size=3, **kwargs).transport_matrix
        assert res[3].nnz == 0
        assert np.all(np.diff(res.indptr)[[0, 1, 2, 4, 5, 6, 7]] >= 1)  # every row with mass survives

    def test_massless_row_does_not_disable_sparsification(self) -> None:
        tmap = self._tmap()
        res = MockSolverOutput(tmap).sparsify(mode="min_row", batch_size=3).transport_matrix
        # without the central policy the threshold collapses to 0 and nothing is sparsified
        assert res.nnz < 0.5 * tmap.size
        with_mass = np.delete(tmap, 3, axis=0)
        np.testing.assert_allclose(
            np.delete(res.toarray(), 3, axis=0),
            np.where(with_mass >= with_mass.max(axis=1).min(), with_mass, 0.0),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_massless_rows_warn(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="moscot"):
            MockSolverOutput(self._tmap()).sparsify(mode="min_row", batch_size=3)
        assert "carry no mass" in caplog.text

    def test_all_rows_massless(self) -> None:
        res = MockSolverOutput(np.zeros((4, 3))).sparsify(mode="min_row", batch_size=2).transport_matrix
        assert res.nnz == 0
        assert res.shape == (4, 3)

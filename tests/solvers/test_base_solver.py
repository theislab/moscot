from typing import Tuple

from scipy.sparse import csr_matrix
import pytest

import numpy as np

from tests._utils import ATOL, RTOL, MockSolverOutput


class TestBaseSolverOutput:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("threshold", [0.0, 1e-1, 1.0])
    @pytest.mark.parametrize("shape", [(7, 2), (91, 103)])
    def test_sparsify(self, batch_size: int, threshold: float, shape: Tuple[int, int]) -> None:
        rng = np.random.RandomState(42)
        tmap = np.abs(rng.rand(shape[0], shape[1]))
        tmap = tmap / tmap.sum()
        output = MockSolverOutput(tmap)
        res = output.sparsify(threshold=threshold, batch_size=batch_size)
        assert isinstance(res, csr_matrix)
        assert res.shape == shape
        assert np.all(res.data >= 0)
        if threshold == 0.0:
            np.testing.assert_allclose(res.A, tmap, rtol=RTOL, atol=ATOL)
        if threshold == 1e-8:
            data = res.data
            assert np.sum((data > threshold) + (data == 0)) == len(data)
        if threshold == 1.0:
            assert np.all(res.data == 0)

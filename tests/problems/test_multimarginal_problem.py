# TODO(@MUCDK) test for correct shape of marginals

from typing import Tuple

import numpy as np

from anndata import AnnData

from moscot.problems import OTProblem
from moscot.solvers._output import BaseSolverOutput


class TestMultiMarginalProblem:
    def test_subclass_OTProblem(self, adata_x: AnnData):
        prob = MockMultiMarginalProblem(adata_x)
        assert isinstance(prob, OTProblem)

    def test_marginal_dtypes(self, adata_x: AnnData, adata_y: AnnData):
        prob = MockMultiMarginalProblem(adata_x, adata_y)
        prob = prob.prepare(xy={"x_attr": "X", "y_attr": "X"})

        assert isinstance(prob._a, list)
        assert isinstance(prob._b, list)
        assert isinstance(prob.a, np.ndarray)
        assert isinstance(prob.b, np.ndarray)

    def test_multiple_iterations(self, adata_x: AnnData, adata_y: AnnData):
        prob = MockMultiMarginalProblem(adata_x, adata_y)
        prob = prob.prepare(xy={"x_attr": "X", "y_attr": "X"})
        prob.solve(n_iters=3)

        assert isinstance(prob.solution, BaseSolverOutput)
        assert prob.a.shape[1] == 4
        assert prob.b.shape[1] == 4
        assert all(prob.a[:, 0] == np.ones(len(adata_x)) / len(adata_x))
        assert all(prob.b[:, 0] == np.ones(len(adata_y)) / len(adata_y))

        last_marginals = prob._get_last_marginals()
        assert isinstance(last_marginals, Tuple)
        assert len(last_marginals[0].shape) == 1
        assert len(last_marginals[1].shape) == 1

    def test_reset_marginals(self, adata_x: AnnData, adata_y: AnnData):
        prob = MockMultiMarginalProblem(adata_x, adata_y)
        prob = prob.prepare(xy={"x_attr": "X", "y_attr": "X"})
        prob.solve(n_iters=1)

        assert prob.a.shape[1] == 2
        assert prob.b.shape[1] == 2

        prob._reset_marginals()

        assert prob.a.shape[1] == 1
        assert prob.b.shape[1] == 1
        assert len(prob._a) == 1
        assert len(prob._b) == 1

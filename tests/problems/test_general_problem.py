from _utils import ATOL, RTOL, Geom_t, TestSolverOutput
import pytest

from ott.geometry import PointCloud
from ott.core.sinkhorn import sinkhorn
import numpy as np
import jax.numpy as jnp

from anndata import AnnData

from moscot.problems import OTProblem
from moscot.solvers._output import BaseSolverOutput


class TestGeneralProblem:
    def test_simple_run(self, adata_x: AnnData, adata_y: AnnData):
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            xy={"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"},
            x={"attr": "X"},
            y={"attr": "X"},
        ).solve(epsilon=5e-1)

        assert isinstance(prob.solution, BaseSolverOutput)

    def test_output(self, adata_x: AnnData, x: Geom_t):
        problem = OTProblem(adata_x)
        problem._solution = TestSolverOutput(x * x.T)

        assert problem.solution.shape == (len(x), len(x))

    @pytest.mark.parametrize("scale_cost", ["max_cost", "max_bound"])
    def test_passing_scale(self, adata_x: AnnData, scale_cost: str):
        scale_cost, online, eps = "max_cost", True, 5e-2
        gt = sinkhorn(PointCloud(jnp.asarray(adata_x.X), online=online, epsilon=eps, scale_cost=scale_cost))

        prob = OTProblem(adata_x)
        prob = prob.prepare(xy={"x_attr": "X", "y_attr": "X"}).solve(online=online, epsilon=eps, scale_cost=scale_cost)
        sol = prob.solution

        np.testing.assert_allclose(gt.matrix, sol.transport_matrix, rtol=RTOL, atol=ATOL)


class MultiMarginalProblem:
    pass

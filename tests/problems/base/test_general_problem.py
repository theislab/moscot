from typing import Literal

import pandas as pd
import pytest

from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear.sinkhorn import sinkhorn
import numpy as np
import jax.numpy as jnp

from anndata import AnnData

from tests._utils import ATOL, RTOL, Geom_t, MockSolverOutput
from moscot.problems.base import OTProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import ProblemKind


class TestOTProblem:
    def test_simple_run(self, adata_x: AnnData, adata_y: AnnData):
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            xy={"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"},
            x={"attr": "X"},
            y={"attr": "X"},
        ).solve(epsilon=5e-1)

        assert isinstance(prob.solution, BaseSolverOutput)

    @pytest.mark.fast()
    def test_output(self, adata_x: AnnData, x: Geom_t):
        problem = OTProblem(adata_x)
        problem._solution = MockSolverOutput(x * x.T)

        assert problem.solution.shape == (len(x), len(x))

    @pytest.mark.parametrize("scale_cost", ["max_cost", "max_bound"])
    def test_passing_scale(self, adata_x: AnnData, scale_cost: str):
        scale_cost, batch_size, eps = "max_cost", 64, 5e-2
        gt = sinkhorn(PointCloud(jnp.asarray(adata_x.X), batch_size=batch_size, epsilon=eps, scale_cost=scale_cost))

        prob = OTProblem(adata_x)
        prob = prob.prepare(xy={"x_attr": "X", "y_attr": "X"}).solve(
            batch_size=batch_size, epsilon=eps, scale_cost=scale_cost
        )
        sol = prob.solution

        np.testing.assert_allclose(gt.matrix, sol.transport_matrix, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("tag", ["cost", "kernel"])
    def test_set_xy(self, adata_x: AnnData, adata_y: AnnData, tag: Literal["cost", "kernel"]):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            xy={"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"},
            x={"attr": "X"},
            y={"attr": "X"},
        )

        cm = rng.uniform(1, 10, size=(adata_x.n_obs, adata_y.n_obs))
        cost_matrix = pd.DataFrame(index=adata_x.obs_names, columns=adata_y.obs_names, data=cm)
        prob.set_xy(cost_matrix, tag=tag)

        prob = prob.solve(max_iterations=5)
        np.testing.assert_equal(prob.xy.data_src, cost_matrix.to_numpy())

    @pytest.mark.parametrize("tag", ["cost", "kernel"])
    def test_set_x(self, adata_x: AnnData, adata_y: AnnData, tag: Literal["cost", "kernel"]):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            xy={"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"},
            x={"attr": "X"},
            y={"attr": "X"},
        )

        cm = rng.uniform(1, 10, size=(adata_x.n_obs, adata_x.n_obs))
        cost_matrix = pd.DataFrame(index=adata_x.obs_names, columns=adata_x.obs_names, data=cm)

        prob = prob.solve(max_iterations=5)
        np.testing.assert_equal(prob.x.data_src, cost_matrix.to_numpy())

    @pytest.mark.parametrize("tag", ["cost", "kernel"])
    def test_set_y(self, adata_x: AnnData, adata_y: AnnData, tag: Literal["cost", "kernel"]):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            xy={"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"},
            x={"attr": "X"},
            y={"attr": "X"},
        )

        cm = rng.uniform(1, 10, size=(adata_y.n_obs, adata_y.n_obs))
        cost_matrix = pd.DataFrame(index=adata_y.obs_names, columns=adata_y.obs_names, data=cm)
        prob.set_y(cost_matrix, tag=tag)

        prob = prob.solve(max_iterations=5)
        np.testing.assert_equal(prob.y.data_src, cost_matrix.to_numpy())

    def test_set_xy_change_problem_kind(self, adata_x: AnnData, adata_y: AnnData):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            x={"attr": "X"},
            y={"attr": "X"},
        )
        assert prob.problem_kind == ProblemKind.QUAD

        cm = rng.uniform(1, 10, size=(adata_x.n_obs, adata_y.n_obs))
        cost_matrix = pd.DataFrame(index=adata_x.obs_names, columns=adata_y.obs_names, data=cm)
        prob.set_xy(cost_matrix, tag="cost")

        assert prob.problem_kind == ProblemKind.QUAD_FUSED

    def test_set_x_change_problem_kind(self, adata_x: AnnData, adata_y: AnnData):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            xy={"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"},
        )
        assert prob.problem_kind == ProblemKind.LINEAR

        cm = rng.uniform(1, 10, size=(adata_x.n_obs, adata_x.n_obs))
        cost_matrix = pd.DataFrame(index=adata_x.obs_names, columns=adata_x.obs_names, data=cm)
        prob.set_x(cost_matrix, tag="cost")

        cm = rng.uniform(1, 10, size=(adata_y.n_obs, adata_y.n_obs))
        cost_matrix = pd.DataFrame(index=adata_y.obs_names, columns=adata_y.obs_names, data=cm)
        prob.set_y(cost_matrix, tag="cost")

        assert prob.problem_kind == ProblemKind.QUAD_FUSED


class MultiMarginalProblem:
    pass

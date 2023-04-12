from typing import Literal

import pytest

import jax.numpy as jnp
import numpy as np
import pandas as pd
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear.sinkhorn import solve as sinkhorn

from anndata import AnnData

from moscot.base.output import BaseSolverOutput, MatrixSolverOutput
from moscot.base.problems import OTProblem
from tests._utils import ATOL, RTOL, Geom_t, MockSolverOutput


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

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_xy(self, adata_x: AnnData, adata_y: AnnData, tag: Literal["cost_matrix", "kernel"]):
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
        assert isinstance(prob.xy.data_src, np.ndarray)
        assert prob.xy.data_tgt is None

        prob = prob.solve(max_iterations=5)  # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
        np.testing.assert_equal(prob.xy.data_src, cost_matrix.to_numpy())

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_x(self, adata_x: AnnData, adata_y: AnnData, tag: Literal["cost_matrix", "kernel"]):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            xy={"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"},
            x={"attr": "X"},
            y={"attr": "X"},
        )

        cm = rng.uniform(1, 10, size=(adata_x.n_obs, adata_x.n_obs))
        cost_matrix = pd.DataFrame(index=adata_x.obs_names, columns=adata_x.obs_names, data=cm)
        prob.set_x(cost_matrix, tag=tag)
        assert isinstance(prob.x.data_src, np.ndarray)
        assert prob.x.data_tgt is None

        prob = prob.solve(max_iterations=5)  # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
        np.testing.assert_equal(prob.x.data_src, cost_matrix.to_numpy())

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_y(self, adata_x: AnnData, adata_y: AnnData, tag: Literal["cost_matrix", "kernel"]):
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
        assert isinstance(prob.y.data_src, np.ndarray)
        assert prob.y.data_tgt is None

        prob = prob.solve(max_iterations=5)
        np.testing.assert_equal(prob.y.data_src, cost_matrix.to_numpy())

    def test_set_xy_change_problem_kind(self, adata_x: AnnData, adata_y: AnnData):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            x={"attr": "X"},
            y={"attr": "X"},
        )
        assert prob.problem_kind == "quadratic"

        cm = rng.uniform(1, 10, size=(adata_x.n_obs, adata_y.n_obs))
        cost_matrix = pd.DataFrame(index=adata_x.obs_names, columns=adata_y.obs_names, data=cm)
        prob.set_xy(cost_matrix, tag="cost_matrix")

        assert prob.problem_kind == "quadratic"

    def test_set_x_change_problem_kind(self, adata_x: AnnData, adata_y: AnnData):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        prob = prob.prepare(
            xy={"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"},
        )
        assert prob.problem_kind == "linear"

        cm = rng.uniform(1, 10, size=(adata_x.n_obs, adata_x.n_obs))
        cost_matrix = pd.DataFrame(index=adata_x.obs_names, columns=adata_x.obs_names, data=cm)
        prob.set_x(cost_matrix, tag="cost_matrix")

        cm = rng.uniform(1, 10, size=(adata_y.n_obs, adata_y.n_obs))
        cost_matrix = pd.DataFrame(index=adata_y.obs_names, columns=adata_y.obs_names, data=cm)
        prob.set_y(cost_matrix, tag="cost_matrix")

        assert prob.problem_kind == "quadratic"

    @pytest.mark.parametrize("clazz", [np.array, pd.DataFrame, MatrixSolverOutput])
    def test_set_solution(self, adata_x: AnnData, adata_y: AnnData, clazz: type):
        rng = np.random.RandomState(42)
        prob = OTProblem(adata_x, adata_y)
        solution = rng.uniform(1, 10, size=prob.shape)
        if clazz is pd.DataFrame:
            solution = clazz(solution, index=prob.adata_src.obs_names, columns=prob.adata_tgt.obs_names)
        elif clazz is MatrixSolverOutput:
            solution = clazz(solution, cost=42, converged=True)
        else:
            solution = clazz(solution)

        prob = prob.set_solution(solution, cost=42, converged=True)

        assert prob.stage == "solved"
        assert isinstance(prob.solution, BaseSolverOutput)
        assert prob.solution.shape == prob.shape
        assert prob.solution.cost == 42
        assert prob.solution.converged

        _ = prob.push()
        _ = prob.pull()

        with pytest.raises(ValueError, match=r".* already contains a solution"):
            _ = prob.set_solution(solution, overwrite=False)

        solution2 = MatrixSolverOutput(solution, cost=42, converged=False)
        prob = prob.set_solution(solution2, overwrite=True)

        assert prob.solution is solution2

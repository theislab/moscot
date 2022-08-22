from typing import Literal, Mapping
import os

from pytest_mock import MockerFixture
from sklearn.metrics.pairwise import euclidean_distances
import pytest

from ott.geometry import PointCloud
from ott.core.sinkhorn import sinkhorn
import jax
import numpy as np
import jax.numpy as jnp

from anndata import AnnData

from tests._utils import ATOL, RTOL, Problem
from moscot.problems.base import OTProblem, CompoundProblem
from moscot.solvers._tagged_array import Tag, TaggedArray


class TestCompoundProblem:
    @staticmethod
    def callback(
        adata: AnnData, adata_y: AnnData, sentinel: bool = False
    ) -> Mapping[Literal["xy", "x", "y"], TaggedArray]:
        assert sentinel
        assert isinstance(adata_y, AnnData)
        return {"xy": TaggedArray(euclidean_distances(adata.X, adata_y.X), tag=Tag.COST_MATRIX)}

    def test_sc_pipeline(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = Problem(adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            key="time",
            axis="obs",
            policy="sequential",
        )
        problem = problem.solve()

        assert len(problem) == len(expected_keys)
        assert isinstance(problem.solutions, dict)
        assert isinstance(problem.problems, dict)
        assert set(problem.solutions.keys()) == set(expected_keys)
        assert set(problem.solutions.keys()) == set(expected_keys)

        for key in problem:
            assert isinstance(problem[key], OTProblem)
            assert problem[key].solution is problem.solutions[key]

    @pytest.mark.fast()
    def test_default_callback(self, adata_time: AnnData, mocker: MockerFixture):
        subproblem = OTProblem(adata_time, adata_y=adata_time.copy())
        callback_kwargs = {"n_comps": 5}
        spy = mocker.spy(subproblem, "_local_pca_callback")

        problem = Problem(adata_time)
        mocker.patch.object(problem, attribute="_create_problem", return_value=subproblem)

        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            key="time",
            axis="obs",
            policy="sequential",
            callback="local-pca",
            callback_kwargs=callback_kwargs,
        )

        assert isinstance(problem, CompoundProblem)
        assert isinstance(problem.problems, dict)
        spy.assert_called_with(subproblem.adata, subproblem._adata_y, **callback_kwargs)

    @pytest.mark.fast()
    def test_custom_callback(self, adata_time: AnnData, mocker: MockerFixture):
        expected_keys = [(0, 1), (1, 2)]
        spy = mocker.spy(TestCompoundProblem, "callback")

        problem = Problem(adata=adata_time)
        _ = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            x={"attr": "X"},
            y={"attr": "X"},
            key="time",
            axis="obs",
            policy="sequential",
            callback=TestCompoundProblem.callback,
            callback_kwargs={"sentinel": True},
        )

        assert spy.call_count == len(expected_keys)

    def test_different_passings_linear(self, adata_with_cost_matrix: AnnData):
        epsilon = 5
        xy = {"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"}
        p1 = Problem(adata_with_cost_matrix)
        p1 = p1.prepare(key="batch", xy=xy)
        p1 = p1.solve(epsilon=epsilon, scale_cost="mean")
        p1_tmap = p1[0, 1].solution.transport_matrix

        p2 = Problem(adata_with_cost_matrix)
        p2 = p2.prepare(key="batch", xy={"attr": "uns", "key": 0, "loss": None, "tag": "cost"})
        p2 = p2.solve(epsilon=epsilon)
        p2_tmap = p2[0, 1].solution.transport_matrix

        gt = sinkhorn(
            PointCloud(
                adata_with_cost_matrix[adata_with_cost_matrix.obs["batch"] == 0].obsm["X_pca"],
                adata_with_cost_matrix[adata_with_cost_matrix.obs["batch"] == 1].obsm["X_pca"],
                epsilon=epsilon,
                scale_cost="mean",
            )
        )

        np.testing.assert_allclose(gt.geom.x, p1[0, 1].solution._output.geom.x, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(gt.geom.y, p1[0, 1].solution._output.geom.y, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            p1[0, 1].solution._output.geom.cost_matrix, gt.geom.cost_matrix, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            p2[0, 1].solution._output.geom.cost_matrix, gt.geom.cost_matrix, rtol=RTOL, atol=ATOL
        )

        np.testing.assert_allclose(gt.matrix, p1_tmap, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(gt.matrix, p2_tmap, rtol=RTOL, atol=ATOL)

    def test_to_dtype(self, adata_with_cost_matrix: AnnData):
        dtype, epsilon = np.float32, 5
        xy = {"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"}
        p = Problem(adata_with_cost_matrix)
        p = p.prepare(key="batch", xy=xy)
        p = p.solve(epsilon=epsilon, scale_cost="mean", dtype=dtype)

        for out in p.solutions.values():
            leaves = [leaf.dtype == dtype for leaf in jax.tree_leaves(out._output) if isinstance(leaf, jnp.ndarray)]
            assert leaves
            assert out.transport_matrix.dtype == dtype
            np.testing.assert_array_equal(leaves, True)

    def test_add_problem(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = Problem(adata=adata_time)
        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            x={"attr": "X"},
            y={"attr": "X"},
            key="time",
            axis="obs",
            policy="sequential",
        )

        assert list(problem.problems.keys()) == expected_keys

        problem = problem.add_problem((0, 2), xy={"x_attr": "X", "y_attr": "X"}, x={"attr": "X"}, y={"attr": "X"})
        assert list(problem.problems.keys()) == expected_keys + [(0, 2)]

    def test_add_created_problem(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = Problem(adata=adata_time)
        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            x={"attr": "X"},
            y={"attr": "X"},
            key="time",
            axis="obs",
            policy="sequential",
        )

        assert list(problem.problems.keys()) == expected_keys

        problem_2 = Problem(adata=adata_time)
        problem_2 = problem_2.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            x={"attr": "X"},
            y={"attr": "X"},
            key="time",
            subset=[(0, 2)],
            axis="obs",
            policy="explicit",
        )
        problem = problem.add_problem((0, 2), problem_2[0, 2])
        assert list(problem.problems.keys()) == expected_keys + [(0, 2)]

    def test_save_load(self, adata_time: AnnData):
        dir_path = "tests/data"
        file_prefix = "test_save_load"
        print(os.getcwd())
        problem = Problem(adata=adata_time)
        problem = problem.prepare(xy={"x_attr": "X", "y_attr": "X"}, key="time")
        problem.save(dir_path=dir_path, file_prefix=file_prefix)

        p = Problem.load(os.path.join(dir_path, f"{file_prefix}_Problem.pkl"))
        assert isinstance(p, Problem)

import os
from typing import Any, Literal, Optional, Tuple

import pytest
from pytest_mock import MockerFixture

import numpy as np
from ott.geometry.costs import Cosine, Euclidean, SqEuclidean
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import solve as sinkhorn
from sklearn.metrics.pairwise import euclidean_distances

from anndata import AnnData

from moscot.base.problems import CompoundProblem, OTProblem
from moscot.utils.tagged_array import Tag, TaggedArray
from tests._utils import ATOL, RTOL, Problem


class TestCompoundProblem:
    @staticmethod
    def xy_callback(
        term: Literal["x", "y", "xy"], adata: AnnData, adata_y: Optional[AnnData] = None, sentinel: bool = False
    ) -> TaggedArray:
        assert sentinel
        assert isinstance(adata_y, AnnData)
        return TaggedArray(euclidean_distances(adata.X, adata_y.X), tag=Tag.COST_MATRIX)

    @staticmethod
    def x_callback(
        term: Literal["x", "y", "xy"], adata: AnnData, adata_y: Optional[AnnData] = None, sentinel: bool = False
    ) -> TaggedArray:
        assert sentinel
        assert isinstance(adata_y, AnnData)
        return TaggedArray(euclidean_distances(adata.X, adata_y.X), tag=Tag.COST_MATRIX)

    @staticmethod
    def y_callback(
        term: Literal["x", "y", "xy"], adata: AnnData, adata_y: Optional[AnnData] = None, sentinel: bool = False
    ) -> TaggedArray:
        assert sentinel
        assert isinstance(adata_y, AnnData)
        return TaggedArray(euclidean_distances(adata.X, adata_y.X), tag=Tag.COST_MATRIX)

    def test_sc_pipeline(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = Problem(adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            key="time",
            policy="sequential",
        )
        problem = problem.solve(max_iterations=2)

        assert len(problem) == len(expected_keys)
        assert isinstance(problem.solutions, dict)
        assert isinstance(problem.problems, dict)
        assert set(problem.solutions.keys()) == set(expected_keys)
        assert set(problem.solutions.keys()) == set(expected_keys)

        for key in problem:
            assert isinstance(problem[key], OTProblem)
            assert problem[key].solution is problem.solutions[key]

    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.fast
    def test_default_callback(self, adata_time: AnnData, mocker: MockerFixture, scale: bool):
        subproblem = OTProblem(adata_time, adata_tgt=adata_time.copy())
        xy_callback_kwargs = {"n_comps": 5, "scale": scale}
        spy = mocker.spy(subproblem, "_local_pca_callback")

        problem = Problem(adata_time)
        mocker.patch.object(problem, attribute="_create_problem", return_value=subproblem)

        problem = problem.prepare(
            key="time",
            policy="sequential",
            xy_callback="local-pca",
            xy_callback_kwargs=xy_callback_kwargs,
        )

        assert isinstance(problem, CompoundProblem)
        assert isinstance(problem.problems, dict)
        spy.assert_called_with("xy", subproblem.adata_src, subproblem.adata_tgt, **xy_callback_kwargs)

    @pytest.mark.fast
    def test_custom_callback_lin(self, adata_time: AnnData, mocker: MockerFixture):
        expected_keys = [(0, 1), (1, 2)]
        spy = mocker.spy(TestCompoundProblem, "xy_callback")

        problem = Problem(adata=adata_time)
        _ = problem.prepare(
            xy={},
            x={},
            y={},
            key="time",
            policy="sequential",
            xy_callback=TestCompoundProblem.xy_callback,
            xy_callback_kwargs={"sentinel": True},
        )

        assert spy.call_count == len(expected_keys)

    @pytest.mark.fast
    def test_custom_callback_quad(self, adata_time: AnnData, mocker: MockerFixture):
        expected_keys = [(0, 1), (1, 2)]
        spy_x = mocker.spy(TestCompoundProblem, "x_callback")
        spy_y = mocker.spy(TestCompoundProblem, "y_callback")

        problem = Problem(adata=adata_time)
        _ = problem.prepare(
            xy={},
            x={},
            y={},
            key="time",
            policy="sequential",
            x_callback=TestCompoundProblem.x_callback,
            y_callback=TestCompoundProblem.y_callback,
            x_callback_kwargs={"sentinel": True},
            y_callback_kwargs={"sentinel": True},
        )

        assert spy_x.call_count == len(expected_keys)
        assert spy_y.call_count == len(expected_keys)

    def test_different_passings_linear(self, adata_with_cost_matrix: AnnData):
        epsilon = 5
        xy = {"x_attr": "obsm", "x_key": "X_pca", "y_attr": "obsm", "y_key": "X_pca"}
        p1 = Problem(adata_with_cost_matrix)
        p1 = p1.prepare(key="batch", xy=xy, policy="sequential")
        p1 = p1.solve(epsilon=epsilon, scale_cost="mean")
        p1_tmap = p1[0, 1].solution.transport_matrix

        p2 = Problem(adata_with_cost_matrix)
        p2 = p2.prepare(
            policy="sequential", key="batch", xy={"attr": "uns", "key": 0, "cost": "custom", "tag": "cost_matrix"}
        )
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

    @pytest.mark.fast
    @pytest.mark.parametrize("cost", [("sq_euclidean", SqEuclidean), ("euclidean", Euclidean), ("cosine", Cosine)])
    def test_prepare_cost(self, adata_time: AnnData, cost: Tuple[str, Any]):
        problem = Problem(adata=adata_time)
        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X", "cost": cost[0]},
            x={"attr": "X", "cost": cost[0]},
            y={"attr": "X", "cost": cost[0]},
            key="time",
            policy="sequential",
        )
        assert isinstance(problem[0, 1].xy.cost, cost[1])
        assert isinstance(problem[0, 1].x.cost, cost[1])
        assert isinstance(problem[0, 1].y.cost, cost[1])

    @pytest.mark.fast
    @pytest.mark.parametrize("cost", [("sq_euclidean", SqEuclidean), ("euclidean", Euclidean), ("cosine", Cosine)])
    def test_prepare_cost_with_callback(self, adata_time: AnnData, cost: Tuple[str, Any]):
        problem = Problem(adata=adata_time)
        problem = problem.prepare(
            xy_callback="local-pca",
            x_callback="local-pca",
            xy={"x_cost": cost[0], "y_cost": cost[0]},
            x={"cost": cost[0]},
            y={"attr": "X", "cost": cost[0]},
            key="time",
            policy="sequential",
        )
        assert isinstance(problem[0, 1].xy.cost, cost[1])
        assert isinstance(problem[0, 1].x.cost, cost[1])
        assert isinstance(problem[0, 1].y.cost, cost[1])

    @pytest.mark.fast
    def test_prepare_different_costs(self, adata_time: AnnData):
        problem = Problem(adata=adata_time)
        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X", "cost": "sq_euclidean"},
            x={"attr": "X", "cost": "euclidean"},
            y={"attr": "X", "cost": "cosine"},
            key="time",
            policy="sequential",
        )
        assert isinstance(problem[0, 1].xy.cost, SqEuclidean)
        assert isinstance(problem[0, 1].x.cost, Euclidean)
        assert isinstance(problem[0, 1].y.cost, Cosine)

    def test_add_problem(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = Problem(adata=adata_time)
        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            x={"attr": "X"},
            y={"attr": "X"},
            key="time",
            policy="sequential",
        )

        assert list(problem.problems.keys()) == expected_keys
        problem2 = Problem(adata=adata_time)
        problem2 = problem2.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            x={"attr": "X"},
            y={"attr": "X"},
            key="time",
            policy="explicit",
            subset=[(0, 2)],
        )
        problem = problem.add_problem((0, 2), problem2[(0, 2)])
        assert list(problem.problems.keys()) == expected_keys + [(0, 2)]

    def test_add_created_problem(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = Problem(adata=adata_time)
        problem = problem.prepare(
            xy={"x_attr": "X", "y_attr": "X"},
            x={"attr": "X"},
            y={"attr": "X"},
            key="time",
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
            policy="explicit",
        )
        problem = problem.add_problem((0, 2), problem_2[0, 2])
        assert list(problem.problems.keys()) == expected_keys + [(0, 2)]

    def test_save_load(self, adata_time: AnnData):
        # TODO(michalk8): refactor this test
        dir_path = "tests/data"
        file_prefix = "test_save_load"
        file = os.path.join(dir_path, f"{file_prefix}_Problem.pkl")
        if os.path.exists(file):
            os.remove(file)
        problem = Problem(adata=adata_time)
        problem = problem.prepare(xy={"x_attr": "X", "y_attr": "X"}, key="time", policy="sequential")
        problem.save(path=file)

        p = Problem.load(file)
        assert isinstance(p, Problem)

    def test_save_load_solved(self, adata_time: AnnData):
        dir_path = "tests/data"
        file_prefix = "test_save_load"
        file = os.path.join(dir_path, f"{file_prefix}_Problem.pkl")
        if os.path.exists(file):
            os.remove(file)
        problem = Problem(adata=adata_time)
        problem = problem.prepare(policy="sequential", xy={"x_attr": "X", "y_attr": "X"}, key="time")
        problem = problem.solve(max_iterations=10)
        problem.save(file)

        p = Problem.load(file)
        assert isinstance(p, Problem)

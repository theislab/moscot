from typing import Type, Optional

from _utils import ATOL, RTOL
from pytest_mock import MockerFixture
from sklearn.metrics.pairwise import euclidean_distances
import pytest

from ott.geometry import PointCloud
from ott.core.sinkhorn import sinkhorn
import numpy as np

from anndata import AnnData

from moscot.problems import CompoundProblem, SingleCompoundProblem
from moscot.backends.ott import FGWSolver, SinkhornSolver
from moscot.solvers._base_solver import OTSolver, ProblemKind
from moscot.solvers._tagged_array import Tag, TaggedArray
from moscot.problems._base_problem import GeneralProblem
from moscot.problems._compound_problem import Callback_t


class TestSingleCompoundProblem:
    @staticmethod
    def callback(
        adata: AnnData, adata_y: Optional[AnnData], problem_kind: ProblemKind, sentinel: bool = False
    ) -> Callback_t:
        assert sentinel
        assert isinstance(adata_y, AnnData)
        return TaggedArray(euclidean_distances(adata.X, adata_y.X), tag=Tag.COST_MATRIX), None

    def test_sc_pipeline(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = SingleCompoundProblem(adata=adata_time, solver=SinkhornSolver())

        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solutions is None

        problem = problem.prepare(
            x={"attr": "X", "tag": Tag.POINT_CLOUD},
            y={"attr": "X", "tag": Tag.POINT_CLOUD},
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
            assert isinstance(problem[key], GeneralProblem)
            assert problem[key].solution is problem.solutions[key]

    @pytest.mark.parametrize("solver_t", [SinkhornSolver, FGWSolver])
    def test_default_callback(self, adata_time: AnnData, solver_t: Type[OTSolver], mocker: MockerFixture):
        subproblem = GeneralProblem(adata_time)  # doesn't matter that it's not a subset
        callback_kwargs = {"n_comps": 5}
        spy = mocker.spy(subproblem, "_prepare_callback")

        problem = SingleCompoundProblem(adata=adata_time, solver=solver_t(), base_problem_type=GeneralProblem)
        mocker.patch.object(problem, attribute="_create_problem", return_value=subproblem)

        problem = problem.prepare(
            x={"attr": "X", "tag": Tag.POINT_CLOUD},
            y={"attr": "X", "tag": Tag.POINT_CLOUD},
            key="time",
            axis="obs",
            policy="sequential",
            callback="pca_local",
            callback_kwargs=callback_kwargs,
        )

        assert isinstance(problem, SingleCompoundProblem)
        assert isinstance(problem.problems, dict)
        spy.assert_called_with(subproblem.adata, subproblem._adata_y, subproblem.solver.problem_kind, **callback_kwargs)

    @pytest.mark.parametrize("solver_t", [SinkhornSolver, FGWSolver])
    def test_custom_callback(self, adata_time: AnnData, mocker: MockerFixture, solver_t: Type[OTSolver]):
        expected_keys = [(0, 1), (1, 2)]
        callback_kwargs = {"sentinel": True}
        spy = mocker.spy(TestSingleCompoundProblem, "callback")

        problem = SingleCompoundProblem(adata=adata_time, solver=solver_t(), base_problem_type=GeneralProblem)
        _ = problem.prepare(
            x={"attr": "X", "tag": Tag.POINT_CLOUD},
            y={"attr": "X", "tag": Tag.POINT_CLOUD},
            key="time",
            axis="obs",
            policy="sequential",
            callback=TestSingleCompoundProblem.callback,
            callback_kwargs=callback_kwargs,
        )

        assert spy.call_count == len(expected_keys)


class TestMultiCompoundProblem:
    pass


class TestCompoundProblem:
    def test_different_passings_linear(self, adata_with_cost_matrix: AnnData):
        epsilon = 5
        x = y = {"attr": "obsm", "key": "X_pca", "tag": "point_cloud"}
        p1 = CompoundProblem(adata_with_cost_matrix, solver=SinkhornSolver())
        p1 = p1.prepare(key="batch", x=x, y=y)
        p1 = p1.solve(epsilon=epsilon, scale_cost="mean")
        p1_tmap = p1[0, 1].solution.transport_matrix

        p2 = CompoundProblem(adata_with_cost_matrix, solver=SinkhornSolver())
        p2 = p2.prepare(key="batch", x={"attr": "uns", "key": 0, "loss": None, "tag": "cost"})
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

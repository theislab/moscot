from typing import Type, Optional

from pytest_mock import MockerFixture
from sklearn.metrics.pairwise import euclidean_distances
import pytest

from anndata import AnnData

from moscot.problems import SingleCompoundProblem
from moscot.backends.ott import FGWSolver, SinkhornSolver
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.solvers._tagged_array import Tag, TaggedArray
from moscot.problems._base_problem import BaseProblem, GeneralProblem
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
        expected_keys = {("0", "1"), ("1", "2")}
        problem = SingleCompoundProblem(adata=adata_time, solver=SinkhornSolver())

        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solution is None

        problem = problem.prepare(
            x={"attr": "X", "tag": Tag.POINT_CLOUD},
            y={"attr": "X", "tag": Tag.POINT_CLOUD},
            key="time",
            axis="obs",
            policy="sequential",
        )
        problem = problem.solve()

        assert len(problem) == len(expected_keys)
        assert isinstance(problem.solution, dict)
        assert isinstance(problem.problems, dict)
        assert set(problem.solution.keys()) == expected_keys
        assert set(problem.solution.keys()) == expected_keys

        for key, subprob in problem:
            assert isinstance(subprob, BaseProblem)
            assert subprob.solution is problem.solution[key]

    @pytest.mark.parametrize("solver_t", [SinkhornSolver, FGWSolver])
    def test_default_callback(self, adata_time: AnnData, solver_t: Type[BaseSolver], mocker: MockerFixture):
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
    def test_custom_callback(self, adata_time: AnnData, mocker: MockerFixture, solver_t: Type[BaseSolver]):
        expected_keys = {("0", "1"), ("1", "2")}
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

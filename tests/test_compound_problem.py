from anndata import AnnData

from moscot.problems import SingleCompoundProblem
from moscot.backends.ott import SinkhornSolver
from moscot.solvers._base_solver import ProblemKind
from moscot.solvers._tagged_array import Tag
from moscot.problems._base_problem import BaseProblem
from moscot.problems._compound_problem import Callback_t


class TestSingleCompoundProblem:
    @staticmethod
    def custom_callback(adata: AnnData, adata_y: AnnData, problem_kind: ProblemKind, **kwargs) -> Callback_t:
        pass

    def test_sc_pipeline(self, adata_time: AnnData):
        expected_keys = {("0", "1"), ("1", "2")}
        solver = SinkhornSolver()
        problem = SingleCompoundProblem(adata=adata_time, solver=solver)

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
        assert expected_keys == set(problem.solution.keys())
        assert expected_keys == set(problem.solution.keys())

        for key, subprob in problem:
            assert isinstance(subprob, BaseProblem)
            assert subprob.solution is problem.solution[key]

    def test_default_callback(self):
        pass

    def test_custom_callback(self):
        pass


class TestMultiCompoundProblem:
    pass

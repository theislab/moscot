#TODO(@MUCDK) test for correct shape of marginals

from typing import Type, Tuple

from _utils import TestSolverOutput
from conftest import ATOL, RTOL, Geom_t
import pytest

from ott.geometry import PointCloud
from ott.core.sinkhorn import sinkhorn
import numpy as np
import jax.numpy as jnp

from anndata import AnnData

from moscot.problems import GeneralProblem
from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import BaseSolver
from moscot.solvers._tagged_array import Tag
from moscot.problems import MultiMarginalProblem
from _utils import MockMultiMarginalProblem


class TestMultiMarginalProblem:
    
    def test_subclass_GeneralProblem(self, adata_x: AnnData):
        prob = MultiMarginalProblem(adata_x, solver=SinkhornSolver())
        prob = prob.prepare(adata_x)
        assert isinstance(prob, GeneralProblem)

    def test_marginal_dtypes(self, adata_x: AnnData, adata_y: AnnData):
        prob = MultiMarginalProblem(adata_x, adata_y, solver=SinkhornSolver())
        prob = prob.prepare(x={"attr": "X"}, y={"attr": "X"})

        assert isinstance(prob._a, list)
        assert isinstance(prob._b, list)
        assert isinstance(prob.a, np.ndarray)
        assert isinstance(prob.b, np.ndarray)

    def test_multiple_iterations(self, adata_x: AnnData, adata_y: AnnData):
        prob = MultiMarginalProblem(adata_x, adata_y, solver=SinkhornSolver())
        prob = prob.prepare(x={"attr": "X"}, y={"attr": "X"})
        prob.solve(n_iters=3)

        assert isinstance(prob.solution, BaseSolverOutput)
        assert prob.a.shape[1] == 4
        assert prob.b.shape[1] == 4
        assert all(prob.a[:,0] == np.ones(len(adata_x))/len(adata_x))
        assert all(prob.b[:,0] == np.ones(len(adata_x))/len(adata_x))
        
        last_marginals = prob._get_last_marginals()
        assert isinstance(last_marginals, Tuple)
        assert len(last_marginals[0].shape) == 1
        assert len(last_marginals[1].shape) == 1

    def test_reset_marginals(self, adata_x: AnnData, adata_y: AnnData):
        prob = MultiMarginalProblem(adata_x, adata_y, solver=SinkhornSolver())
        prob = prob.prepare(x={"attr": "X"}, y={"attr": "X"})
        prob.solve(n_iters=1)

        assert prob.a.shape[1] == 2
        assert prob.b.shape[1] == 2

        prob._reset_marginals()

        assert prob.a.shape[1] == 1
        assert prob.b.shape[1] == 1
        assert len(prob._a) == 1
        assert len(prob._b) == 1

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Type, Tuple, Mapping, Optional

import numpy.typing as npt

from anndata import AnnData

from moscot.tmp.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.tmp.solvers._data import TaggedArray
from moscot.tmp.solvers._output import BaseSolverOutput
from moscot.tmp.problems._anndata import AnnDataPointer
from moscot.tmp.solvers._base_solver import BaseSolver


class BaseProblem(ABC):
    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None):
        # TODO(michalk8): check if view, if yes, materialize
        self._adata = adata
        self._solver: BaseSolver = self._create_default_solver(solver)

    @property
    @abstractmethod
    def _valid_solver_types(self) -> Tuple[Type[BaseSolver], ...]:
        # TODO(michalk8): backend dependend
        pass

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        pass

    # TODO(michalk8): figure out args/kwargs
    @abstractmethod
    def solve(self) -> "BaseProblem":
        pass

    # usually, these will be provided by mixins
    @abstractmethod
    def push_forward(self, x: npt.ArrayLike, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def pull_backward(self, x: npt.ArrayLike, **kwargs) -> Any:
        pass

    def _create_default_solver(self, solver: Optional[BaseSolver] = None) -> BaseSolver:
        if not len(self._valid_solver_types):
            raise ValueError("TODO: shouldn't happen")
        if solver is not None:
            if not isinstance(solver, self._valid_solver_types):
                raise TypeError("TODO: wrong type")
            return solver
        # TODO(michalk8): this assumes all solvers always have defaults
        # alt. would be to force a classmethod called e.g. `BaseSolver.create(cls)`
        return self._valid_solver_types[0]()

    # TODO(michalk8): add IO (de)serialization from/to adata?

    @property
    def adata(self) -> AnnData:
        return self.adata

    @property
    @abstractmethod
    # endpoint for mixins, will be used to check whether downstream methods can run
    def solution(self) -> Any:  # Optional[Union[SolverOutput, Mapping[..., SolverOutput]]]?
        pass

    # TODO(michalk8): backed/solvers getters/setters? must be coupled


class GeneralProblem(BaseProblem):
    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None):
        super().__init__(adata, solver)
        self._solution: Optional[BaseSolverOutput] = None
        self._x: Optional[TaggedArray] = None
        self._y: Optional[TaggedArray] = None
        self._xy: Optional[TaggedArray] = None

    @property
    def _valid_solver_types(self) -> Tuple[Type[BaseSolver], ...]:
        return SinkhornSolver, GWSolver, FGWSolver

    def prepare(
        self,
        x: Mapping[str, Any] = MappingProxyType({}),
        y: Mapping[str, Any] = MappingProxyType({}),
        xy: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "BaseProblem":
        self._x = AnnDataPointer.from_dict(x).create(**kwargs)
        self._y = AnnDataPointer.from_dict(y).create(**kwargs)
        self._xy = AnnDataPointer.from_dict(xy).create(**kwargs)

    def solve(self, **kwargs: Any) -> "BaseProblem":
        self._solution = self._solver(self._x, self._y, self._xy, **kwargs)
        return self

    @property
    def solution(self) -> BaseSolverOutput:
        return self._solution

    # TODO(michalk8): do we need the kwargs in base class?
    def push_forward(self, x: npt.ArrayLike, **_: Any) -> npt.ArrayLike:
        return self.solution.push_forward(x)

    def pull_backward(self, x: npt.ArrayLike, **_: Any) -> npt.ArrayLike:
        return self.solution.pull_backward(x)

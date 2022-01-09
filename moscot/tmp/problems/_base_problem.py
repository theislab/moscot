from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Mapping, Optional

import numpy.typing as npt

from anndata import AnnData

from moscot.tmp.backends.ott import GWSolver, FGWSolver, SinkhornSolver
from moscot.tmp.solvers._data import Tag, TaggedArray
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
        return self._adata

    @property
    @abstractmethod
    # endpoint for mixins, will be used to check whether downstream methods can run
    def solution(self) -> Any:  # Optional[Union[SolverOutput, Mapping[..., SolverOutput]]]?
        pass

    # TODO(michalk8): backed/solvers getters/setters? must be coupled


class GeneralProblem(BaseProblem):
    def __init__(
        self,
        adata_x: AnnData,
        adata_y: Optional[AnnData] = None,
        adata_xy: Optional[AnnData] = None,
        solver: Optional[BaseSolver] = None,
    ):
        super().__init__(adata_x, solver)
        self._solution: Optional[BaseSolverOutput] = None

        self._x: Optional[TaggedArray] = None
        self._y: Optional[TaggedArray] = None
        self._xy: Optional[Union[TaggedArray, TaggedArray]] = None

        self._adata_y = adata_y
        self._adata_xy = adata_xy

        if adata_xy is not None:
            if adata_y is None:
                raise ValueError("TODO: adata_y must be present if adata_xy is present")
            if adata_x.n_obs != adata_xy.n_obs:
                raise ValueError("First and joint shape mismatch")
            if adata_y.n_obs != adata_xy.n_vars:
                raise ValueError("First and joint shape mismatch")

    @property
    def _valid_solver_types(self) -> Tuple[Type[BaseSolver], ...]:
        return SinkhornSolver, GWSolver, FGWSolver

    def _handle_joint(
        self, tag: Tag = Tag.COST, create_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any
    ) -> Optional[Union[TaggedArray, Tuple[TaggedArray, TaggedArray]]]:
        if not isinstance(self._solver, FGWSolver):
            return None

        tag = Tag(tag)
        if tag in (Tag.COST, Tag.KERNEL):
            attr = kwargs.get("attr", "X")
            if attr == "obsm":
                return AnnDataPointer(self.adata, tag=tag, **kwargs).create(**create_kwargs)
            if attr == "varm":
                kwargs["attr"] = "obsm"
                return AnnDataPointer(self._adata_y.T, tag=tag, **kwargs).create(**create_kwargs)
            if attr not in ("X", "layers", "raw"):
                raise AttributeError("TODO: expected obsm/varm/X/layers/raw")
            if self._adata_xy is None:
                raise ValueError("TODO: Specifying cost/kernel requires joint adata.")
            return AnnDataPointer(self._adata_xy, tag=tag, **kwargs).create(**create_kwargs)
        if tag != Tag.POINT_CLOUD:
            # TODO(michalk8): warn
            tag = Tag.POINT_CLOUD

        x_kwargs = {k: v for k, v in kwargs.items() if k.startswith("x_")}
        y_kwargs = {k: v for k, v in kwargs.items() if k.startswith("y_")}

        x_array = AnnDataPointer(self.adata, tag=tag, **x_kwargs).create(**create_kwargs)
        y_array = AnnDataPointer(self.adata, tag=tag, **y_kwargs).create(**create_kwargs)

        return x_array, y_array

    def prepare(
        self,
        x: Mapping[str, Any] = MappingProxyType({}),
        y: Optional[Mapping[str, Any]] = None,
        xy: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "BaseProblem":
        self._x = AnnDataPointer(adata=self.adata, **x).create(**kwargs)
        self._y = None if y is None else AnnDataPointer(adata=self._adata_y, **y).create(**kwargs)
        self._xy = None if xy is None else self._handle_joint(**xy, create_kwargs=kwargs)

        return self

    def solve(self, **kwargs: Any) -> "BaseProblem":
        if not isinstance(self._solver, FGWSolver):
            kwargs["xx"] = kwargs["yy"] = None
        elif isinstance(self._xy, tuple):
            # point cloud
            kwargs["xx"] = self._xy[0]
            kwargs["yy"] = self._xy[1]
        else:
            # cost/kernel
            kwargs["xx"] = self._xy
            kwargs["yy"] = None

        self._solution = self._solver(self._x, self._y, **kwargs)
        return self

    def push_forward(self, x: npt.ArrayLike, **_: Any) -> npt.ArrayLike:
        return self.solution.push_forward(x)

    def pull_backward(self, x: npt.ArrayLike, **_: Any) -> npt.ArrayLike:
        return self.solution.pull_backward(x)

    @property
    def solution(self) -> Optional[BaseSolverOutput]:
        return self._solution

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, List, Tuple, Union, Mapping, Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.backends.ott import SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.problems._anndata import AnnDataPointer
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.solvers._tagged_array import Tag, TaggedArray


class BaseProblem(ABC):
    def __init__(
        self,
        adata: AnnData,
        solver: Optional[BaseSolver] = None,
    ):
        self._adata = adata
        self.solver = self._default_solver if solver is None else solver

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        pass

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        pass

    @property
    @abstractmethod
    def solution(self) -> Optional[BaseSolverOutput]:
        pass

    @property
    @abstractmethod
    def _default_solver(self) -> BaseSolver:
        pass

    @staticmethod
    def _get_mass(
        adata: AnnData,
        data: Optional[Union[str, List[str], Tuple[str, ...], npt.ArrayLike]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
    ) -> npt.ArrayLike:
        if data is None:
            data = np.ones((adata.n_obs,), dtype=float)
        elif isinstance(data, (str, list, tuple)):
            if isinstance(data, (list, tuple)) and not len(data):
                raise ValueError("TODO: no subset keys specified.")
            # TODO: allow mix numeric/categorical keys (how to handle multiple subsets then?)
            if subset is None:  # allow for numeric values
                data = np.asarray(adata.obs[data], dtype=float)
            elif isinstance(subset, Iterable) and not isinstance(subset, str):
                data = np.asarray(adata.obs[data].isin(subset), dtype=float)
            else:
                data = np.asarray(adata.obs[data] == subset, dtype=float)
        else:
            data = np.asarray(data)

        if data.ndim != 2:
            data = np.reshape(data, (-1, 1))
        if data.shape[0] != adata.n_obs:
            raise ValueError(f"TODO: expected shape `{adata.n_obs,}`, found `{data.shape[0],}`")

        total = np.sum(data != 0, axis=0)[None, :]
        if not np.all(total > 0):
            raise ValueError("TODO: no mass.")

        return data / total if normalize else data

    @staticmethod
    def _get_or_create_marginal(adata: AnnData, data: Optional[Union[str, npt.ArrayLike]] = None) -> npt.ArrayLike:
        if data is None:
            return np.ones((adata.n_obs,), dtype=float) / adata.n_obs
        if isinstance(data, str):
            # TODO(michalk8): some nice error message
            data = adata.obs[data]
        data = np.asarray(data)
        # TODO(michalk8): check shape
        return data

    @property
    def adata(self) -> AnnData:
        return self._adata

    @property
    def solver(self) -> BaseSolver:
        return self._solver

    @solver.setter
    def solver(self, solver: BaseSolver) -> None:
        if not isinstance(solver, BaseSolver):
            raise TypeError("TOOD: not a solver")
        self._solver = solver


class GeneralProblem(BaseProblem):
    def __init__(
        self,
        adata_x: AnnData,
        adata_y: Optional[AnnData] = None,
        adata_xy: Optional[AnnData] = None,
        solver: Optional[BaseSolver] = None,
    ):
        super().__init__(adata_x, solver=solver)
        self._solution: Optional[BaseSolverOutput] = None

        self._x: Optional[TaggedArray] = None
        self._y: Optional[TaggedArray] = None
        self._xy: Optional[Union[TaggedArray, TaggedArray]] = None

        self._a: Optional[npt.ArrayLike] = None
        self._b: Optional[npt.ArrayLike] = None

        self._adata_y = adata_y
        self._adata_xy = adata_xy

        if adata_xy is not None:
            if adata_y is None:
                raise ValueError("TODO: adata_y must be present if adata_xy is present")
            if adata_x.n_obs != adata_xy.n_obs:
                raise ValueError("First and joint shape mismatch")
            if adata_y.n_obs != adata_xy.n_vars:
                raise ValueError("First and joint shape mismatch")

    def _handle_joint(
        self, create_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any
    ) -> Union[TaggedArray, Tuple[TaggedArray, TaggedArray]]:
        tag = kwargs.get("tag", None)
        if tag is None:
            # TODO(michalk8): better/more strict condition?
            # TODO(michalk8): specify which tag is being using
            tag = Tag.POINT_CLOUD if "x_attr" in kwargs and "y_attr" in kwargs else Tag.COST_MATRIX

        tag = Tag(tag)
        if tag in (Tag.COST_MATRIX, Tag.KERNEL):
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
            # TODO(michalk8): log-warn
            tag = Tag.POINT_CLOUD

        # TODO(michalk8): mb. be less stringent and assume without the prefix x_ belong to x
        x_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("x_")}
        y_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("y_")}

        x_array = AnnDataPointer(self.adata, tag=tag, **x_kwargs).create(**create_kwargs)
        y_array = AnnDataPointer(self._adata_y, tag=tag, **y_kwargs).create(**create_kwargs)

        return x_array, y_array

    def prepare(
        self,
        x: Mapping[str, Any] = MappingProxyType({}),
        y: Optional[Mapping[str, Any]] = None,
        xy: Optional[Mapping[str, Any]] = None,
        a: Optional[Union[str, npt.ArrayLike]] = None,
        b: Optional[Union[str, npt.ArrayLike]] = None,
        **kwargs: Any,
    ) -> "GeneralProblem":
        self._x = AnnDataPointer(adata=self.adata, **x).create(**kwargs)
        self._y = None if y is None else AnnDataPointer(adata=self._adata_y, **y).create(**kwargs)
        self._xy = (
            None
            if xy is None or self.solver.problem_kind != ProblemKind.QUAD_FUSED
            else self._handle_joint(**xy, create_kwargs=kwargs)
        )

        self._a = self._get_or_create_marginal(self.adata, a)
        self._b = self._get_or_create_marginal(self._marginal_b_adata, b)
        self._solution = None

        return self

    def solve(
        self,
        epsilon: Optional[float] = None,
        **kwargs: Any,
    ) -> "GeneralProblem":
        if isinstance(self._xy, tuple):  # point cloud
            kwargs["xx"] = self._xy[0]
            kwargs["yy"] = self._xy[1]
        else:  # cost/kernel
            kwargs["xx"] = self._xy
            kwargs["yy"] = None

        # this allows for MultiMarginalProblem to pass new marginals
        a = kwargs.pop("a", self._a)
        b = kwargs.pop("b", self._b)

        self._solution = self.solver(self._x, self._y, a=a, b=b, epsilon=epsilon, **kwargs)
        return self

    # TODO(michalk8): require in BaseProblem?
    def push(
        self,
        data: Optional[Union[str, npt.ArrayLike]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> npt.ArrayLike:
        # TODO: check if solved - decorator?
        data = self._get_mass(self.adata, data=data, subset=subset, normalize=normalize)
        return self.solution.push(data, **kwargs)

    def pull(
        self,
        data: Optional[Union[str, npt.ArrayLike]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> npt.ArrayLike:
        # TODO: check if solved - decorator?
        adata = self.adata if self._adata_y is None else self._adata_y
        data = self._get_mass(adata, data=data, subset=subset, normalize=normalize)
        return self.solution.pull(data, **kwargs)

    @property
    def _default_solver(self) -> BaseSolver:
        return SinkhornSolver()

    @property
    def solution(self) -> Optional[BaseSolverOutput]:
        return self._solution

    @property
    def _marginal_b_adata(self) -> AnnData:
        return self.adata if self._adata_y is None else self._adata_y

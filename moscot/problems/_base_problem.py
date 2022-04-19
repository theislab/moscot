from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, List, Tuple, Union, Mapping, Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt

from anndata import AnnData
import scanpy as sc

from moscot.backends.ott import SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.problems._anndata import AnnDataPointer
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ("BaseProblem", "GeneralProblem")


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
                data = np.asarray(adata.obs[data].values == subset, dtype=float)
        else:
            data = np.asarray(data)

        if data.ndim != 2:
            data = np.reshape(data, (-1, 1))
        if data.shape[0] != adata.n_obs:
            raise ValueError(f"TODO: expected shape `{adata.n_obs,}`, found `{data.shape[0],}`")
        if not np.all(data >= 0):
            raise ValueError("Not all entries of the mass are non-negative")
        total = np.sum(data, axis=0)[None, :]
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
        if not isinstance(solver, BaseSolver):  # TODO: enable
            raise TypeError("TOOD: not a solver")
        self._solver = solver


class GeneralProblem(BaseProblem):
    def __init__(
        self,
        adata_x: AnnData,
        adata_y: Optional[AnnData] = None,
        solver: Optional[BaseSolver] = None,
        source: Any = "src",
        target: Any = "tgt",
        **kwargs: Any,
    ):
        super().__init__(adata_x, solver=solver)
        # TODO(michalk8): consider setting this to `adata_x` if None
        self._adata_y = adata_y
        self._solution: Optional[BaseSolverOutput] = None

        self._x: Optional[TaggedArray] = None
        self._y: Optional[TaggedArray] = None
        self._xy: Optional[Union[TaggedArray, TaggedArray]] = None

        self._a: Optional[npt.ArrayLike] = None
        self._b: Optional[npt.ArrayLike] = None

        self._source = source
        self._target = target

    def _handle_joint(self, tag: Optional[Tag] = None, **kwargs) -> Union[TaggedArray, Tuple[TaggedArray, TaggedArray]]:
        if tag is None:
            # TODO(michalk8): better/more strict condition?
            # TODO(michalk8): specify which tag is being using
            tag = Tag.POINT_CLOUD if "x_attr" in kwargs and "y_attr" in kwargs else Tag.COST_MATRIX

        tag = Tag(tag)
        if tag in (Tag.COST_MATRIX, Tag.KERNEL):
            attr = kwargs.get("attr", "obsm")
            if attr == "obsm":
                return AnnDataPointer(self.adata, tag=tag, **kwargs).create()
            if attr == "varm":
                kwargs["attr"] = "obsm"
                return AnnDataPointer(self._adata_y.T, tag=tag, **kwargs).create()
            raise NotImplementedError("TODO: cost/kernel storage not implemented. Use obsm/varm")
        if tag != Tag.POINT_CLOUD:
            # TODO(michalk8): log-warn
            tag = Tag.POINT_CLOUD

        # TODO(michalk8): mb. be less stringent and assume without the prefix x_ belong to x
        x_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("x_")}
        y_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("y_")}

        x_array = AnnDataPointer(self.adata, tag=tag, **x_kwargs).create()
        y_array = AnnDataPointer(self._adata_y, tag=tag, **y_kwargs).create()

        return x_array, y_array

    def prepare(
        self,
        x: Union[TaggedArray, Mapping[str, Any]] = MappingProxyType({}),
        y: Optional[Union[TaggedArray, Mapping[str, Any]]] = None,
        xy: Optional[Union[Tuple[TaggedArray, TaggedArray], Mapping[str, Any]]] = None,
        a: Optional[Union[str, npt.ArrayLike]] = None,
        b: Optional[Union[str, npt.ArrayLike]] = None,
        **_: Any,
    ) -> "GeneralProblem":
        def update_key(kwargs: Mapping[str, Any], *, is_source: bool) -> Mapping[str, Any]:
            if kwargs.get("attr", None) == "uns":
                kwargs = dict(kwargs)
                kwargs["key"] = self._source if is_source else self._target
            return kwargs

        self._x = (
            x
            if isinstance(x, TaggedArray)
            else AnnDataPointer(adata=self.adata, **update_key(x, is_source=True)).create()
        )
        self._y = (
            y
            if y is None or isinstance(y, TaggedArray)
            else AnnDataPointer(adata=self._adata_y, **update_key(y, is_source=False)).create()
        )
        if self.solver.problem_kind != ProblemKind.QUAD_FUSED:
            self._xy = None
        else:
            self._xy = xy if xy is None or isinstance(xy, tuple) else self._handle_joint(**xy)

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

    @staticmethod
    def _prepare_callback(
        adata: AnnData,
        adata_y: Optional[AnnData] = None,
        problem_kind: ProblemKind = ProblemKind.LINEAR,
        layer: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[TaggedArray, Optional[TaggedArray]]:
        n = adata.n_obs
        if problem_kind not in (ProblemKind.LINEAR, ProblemKind.QUAD_FUSED):
            raise NotImplementedError("TODO: invalid problem type")
        adata = adata if adata_y is None else adata.concatenate(adata_y)
        data = adata.X if layer is None else adata.layers[layer]
        data = sc.pp.pca(data, **kwargs)

        return TaggedArray(data[:n], tag=Tag.POINT_CLOUD), TaggedArray(data[n:], tag=Tag.POINT_CLOUD)

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.adata), len(self._marginal_b_adata)

    @property
    def _default_solver(self) -> BaseSolver:
        return SinkhornSolver()

    @property
    def solution(self) -> Optional[BaseSolverOutput]:
        return self._solution

    @property
    def _marginal_b_adata(self) -> AnnData:
        return self.adata if self._adata_y is None else self._adata_y

    @property
    def x(self) -> Optional[npt.ArrayLike]:
        if self._x is None:
            return None
        return self._x.data
        
    @property
    def y(self) -> Optional[npt.ArrayLike]:
        if self._y is None:
            return None
        return self._y.data
        
    @property
    def xy(self) -> Optional[Tuple[npt.ArrayLike, npt.ArrayLike]]:
        if self._xy is None:
            return None
        return self._xy[0].data, self._xy[1].data

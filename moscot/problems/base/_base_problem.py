from abc import ABC, abstractmethod
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Union, Mapping, Iterable, Optional, Sequence

from scipy.sparse import vstack, issparse, csr_matrix
from typing_extensions import Literal

import numpy as np

from anndata import AnnData
import scanpy as sc

from moscot._docs import d
from moscot._types import ArrayLike
from moscot.problems._utils import wrap_solve, wrap_prepare, require_solution
from moscot.solvers._output import BaseSolverOutput
from moscot.problems._anndata import AnnDataPointer
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ["BaseProblem", "OTProblem"]


class ProblemStage(str, Enum):
    INITIALIZED = "initialized"
    PREPARED = "prepared"
    SOLVED = "solved"


@d.get_sections(base="BaseProblem", sections=["Parameters", "Raises"])
@d.dedent
class BaseProblem(ABC):
    """
    Problem base class handling one optimal transport subproblem.

    Parameters
    ----------
    %(adata)s

    Raises
    ------
    ValueError
        If `adata` has no observations.
    ValueError
        If `adata` has no variables.
    """

    def __init__(self, adata: AnnData, copy: bool = False):
        self._adata = adata.copy() if copy else adata
        self._problem_kind: ProblemKind = ProblemKind.UNKNOWN
        self._stage = ProblemStage.INITIALIZED

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        pass

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        pass

    # TODO(michalk8): move below?
    @staticmethod
    def _get_mass(
        adata: AnnData,
        data: Optional[Union[str, List[str], Tuple[str, ...], ArrayLike]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
    ) -> ArrayLike:
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

    @property
    @d.dedent
    def adata(self) -> AnnData:
        """%(adata)s"""  # noqa
        return self._adata

    @property
    def stage(self) -> ProblemStage:  # noqa: D102
        return self._stage

    @property
    def problem_kind(self) -> ProblemKind:  # noqa: D102
        return self._problem_kind


@d.get_sections(base="OTProblem", sections=["Parameters", "Raises"])
@d.dedent
class OTProblem(BaseProblem):
    """
    Problem class handling one optimal transport subproblem.

    Parameters
    ----------
    %(adata_x)s
    %(adata_y)s
    %(source)s
    %(target)s

    Raises
    ------
     %(BaseProblem.raises)s
    """

    def __init__(
        self,
        adata_x: AnnData,
        adata_y: Optional[AnnData] = None,
        *,
        copy: bool = False,
    ):
        super().__init__(adata_x, copy=copy)
        self._adata_y = adata_x if adata_y is None else adata_y.copy() if copy else adata_y
        self._solution: Optional[BaseSolverOutput] = None

        self._x: Optional[TaggedArray] = None
        self._y: Optional[TaggedArray] = None
        self._xy: Optional[TaggedArray] = None

        self._a: Optional[ArrayLike] = None
        self._b: Optional[ArrayLike] = None

    def _handle_linear(self, **kwargs: Any) -> TaggedArray:
        if "x_attr" not in kwargs or "y_attr" not in kwargs:
            kwargs.setdefault("tag", Tag.COST_MATRIX)
            attr = kwargs.pop("attr", "obsm")
            if attr in ("obsm", "uns"):
                return AnnDataPointer(self.adata, attr=attr, **kwargs).create()
            if attr == "varm":
                return AnnDataPointer(self._adata_y.T, attr="obsm", **kwargs).create()
            raise NotImplementedError("TODO: cost/kernel storage not implemented. Use obsm/varm")

        x_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("x_")}
        y_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("y_")}
        x_kwargs["tag"] = Tag.POINT_CLOUD
        y_kwargs["tag"] = Tag.POINT_CLOUD

        # TODO(michalk8): this is legacy creation, adapt
        x_array = AnnDataPointer(self.adata, **x_kwargs).create()
        y_array = AnnDataPointer(self._adata_y, **y_kwargs).create()

        return TaggedArray(x_array.data, y_array.data, tag=Tag.POINT_CLOUD, loss=x_array.loss)

    @wrap_prepare
    def prepare(  # type: ignore[override]
        self,
        xy: Optional[Union[Mapping[str, Any], TaggedArray]] = None,
        x: Optional[Union[Mapping[str, Any], TaggedArray]] = None,
        y: Optional[Union[Mapping[str, Any], TaggedArray]] = None,
        a: Optional[Union[bool, str, ArrayLike]] = None,
        b: Optional[Union[bool, str, ArrayLike]] = None,
        **kwargs: Any,
    ) -> "OTProblem":
        self._x = self._y = self._xy = self._solution = None
        # TODO(michalk8): handle again TaggedArray?
        # TODO(michalk8): better dispatch

        # fmt: off
        if xy is not None and x is None and y is None:
            self._problem_kind = ProblemKind.LINEAR
            self._xy = xy if isinstance(xy, TaggedArray) else self._handle_linear(**xy)
        elif x is not None and y is not None and xy is None:
            self._problem_kind = ProblemKind.QUAD
            self._x = x if isinstance(x, TaggedArray) else AnnDataPointer(adata=self.adata, **x).create()
            self._y = y if isinstance(y, TaggedArray) else AnnDataPointer(adata=self._adata_y, **y).create()
        elif xy is not None and x is not None and y is not None:
            self._problem_kind = ProblemKind.QUAD_FUSED
            self._xy = xy if isinstance(xy, TaggedArray) else self._handle_linear(**xy)
            self._x = x if isinstance(x, TaggedArray) else AnnDataPointer(adata=self.adata, **x).create()
            self._y = y if isinstance(y, TaggedArray) else AnnDataPointer(adata=self._adata_y, **y).create()
        else:
            raise NotImplementedError("TODO: Combination not implemented")
        # fmt: on

        self._a = self._create_marginals(self.adata, data=a, source=True, **kwargs)
        self._b = self._create_marginals(self._adata_y, data=b, source=False, **kwargs)
        return self

    @wrap_solve
    def solve(  # type: ignore[override]
        self,
        epsilon: Optional[float] = 1e-2,
        alpha: Optional[float] = 0.5,
        rank: int = -1,
        scale_cost: Optional[Union[float, str]] = None,
        online: Optional[int] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        prepare_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "OTProblem":
        if self._problem_kind is None:
            raise RuntimeError("Run .prepare() first.")
        if self._problem_kind in (ProblemKind.QUAD, ProblemKind.QUAD_FUSED):
            kwargs["epsilon"] = epsilon
        kwargs["rank"] = rank
        # allow `MultiMarginalProblem` to pass new marginals
        a = kwargs.pop("a", self._a)
        b = kwargs.pop("b", self._b)

        prepare_kwargs = dict(prepare_kwargs)
        prepare_kwargs["epsilon"] = epsilon
        prepare_kwargs["alpha"] = alpha
        prepare_kwargs["scale_cost"] = scale_cost
        prepare_kwargs["online"] = online

        solver: BaseSolver[BaseSolverOutput] = self._problem_kind.solver(backend="ott", **kwargs)
        self._solution = solver(x=self._x, y=self._y, xy=self._xy, a=a, b=b, tau_a=tau_a, tau_b=tau_b, **prepare_kwargs)
        return self

    @require_solution
    def push(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> ArrayLike:
        """Push mass."""
        data = self._get_mass(self.adata, data=data, subset=subset, normalize=normalize)
        return self.solution.push(data, **kwargs)  # type: ignore[union-attr]

    @require_solution
    def pull(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> ArrayLike:
        """Pull mass."""
        adata = self.adata if self._adata_y is None else self._adata_y
        data = self._get_mass(adata, data=data, subset=subset, normalize=normalize)
        return self.solution.pull(data, **kwargs)  # type: ignore[union-attr]

    @staticmethod
    def _local_pca_callback(
        adata: AnnData,
        adata_y: AnnData,
        layer: Optional[str] = None,
        return_linear: bool = True,
        **kwargs: Any,
    ) -> Dict[Literal["xy", "x", "y"], TaggedArray]:
        def concat(x: ArrayLike, y: ArrayLike) -> ArrayLike:
            if issparse(x):
                return vstack([x, csr_matrix(y)])
            if issparse(y):
                return vstack([csr_matrix(x), y])
            return np.vstack([x, y])

        if adata is adata_y:
            raise ValueError(f"TODO: `{adata}`, `{adata_y}`")
        x = adata.X if layer is None else adata.layers[layer]
        y = adata_y.X if layer is None else adata_y.layers[layer]

        if return_linear:
            n = x.shape[0]
            joint_space = kwargs.pop("joint_space", True)
            if joint_space:
                data = sc.pp.pca(concat(x, y), **kwargs)
            else:
                data = concat(sc.pp.pca(x, **kwargs), sc.pp.pca(y, **kwargs))

            return {"xy": TaggedArray(data[:n], data[n:], tag=Tag.POINT_CLOUD)}

        x = sc.pp.pca(x, **kwargs)
        y = sc.pp.pca(y, **kwargs)
        return {"x": TaggedArray(x, tag=Tag.POINT_CLOUD), "y": TaggedArray(y, tag=Tag.POINT_CLOUD)}

    def _create_marginals(
        self, adata: AnnData, *, source: bool, data: Optional[Union[bool, str, ArrayLike]] = None, **kwargs: Any
    ) -> ArrayLike:
        if data is True:
            return self._estimate_marginals(adata, source=source, **kwargs)
        if data in (False, None):
            return np.ones((adata.n_obs,), dtype=float) / adata.n_obs
        if isinstance(data, str):
            # TODO(michalk8): some nice error message
            return np.asarray(adata.obs[data])
        return np.asarray(data)

    def _estimate_marginals(self, adata: AnnData, *, source: bool, **kwargs: Any) -> ArrayLike:
        return np.ones((adata.n_obs,), dtype=float) / adata.n_obs

    @property
    def shape(self) -> Tuple[int, int]:  # noqa: D102
        return self.adata.n_obs, self._adata_y.n_obs

    @property
    def solution(self) -> Optional[BaseSolverOutput]:  # noqa: D102
        return self._solution

    @property
    def x(self) -> Optional[TaggedArray]:  # noqa: D102
        return self._x

    @property
    def y(self) -> Optional[TaggedArray]:  # noqa: D102
        return self._y

    # TODO(michalk8): verify type
    @property
    def xy(self) -> Optional[TaggedArray]:  # noqa: D102
        return self._xy

    @property
    def a(self) -> Optional[ArrayLike]:  # noqa: D102
        return self._a

    @property
    def b(self) -> Optional[ArrayLike]:  # noqa: D102
        return self._b

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[shape={self.adata.n_obs, self._adata_y.n_obs}]"

    def __str__(self) -> str:
        return repr(self)

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Union, Literal, Mapping, Optional, TYPE_CHECKING

from scipy.sparse import vstack, issparse, csr_matrix

import numpy as np

from anndata import AnnData
import scanpy as sc

from moscot._types import ArrayLike, ScaleCost_t, Initializer_t
from moscot._logging import logger
from moscot._docs._docs import d
from moscot.problems._utils import wrap_solve, wrap_prepare, require_solution
from moscot.solvers._output import BaseSolverOutput
from moscot.problems._anndata import AnnDataPointer
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot._constants._constants import ScaleCost, ProblemStage
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ["BaseProblem", "OTProblem", "ProblemKind"]


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
        """Abstract prepare method."""

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        """Abstract solve method."""

    # TODO(michalk8): move below?
    @staticmethod
    def _get_mass(
        adata: AnnData,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        normalize: bool = True,
        *,
        split_mass: bool = False,
    ) -> ArrayLike:
        def _split_mass(arr: ArrayLike) -> ArrayLike:
            if arr.ndim == 2:
                return arr
            non_zero_idxs = arr.nonzero()[0]
            data = np.zeros((len(arr), len(non_zero_idxs)))
            data[non_zero_idxs, np.arange(len(non_zero_idxs))] = arr[non_zero_idxs]
            return data

        if data is None:
            if subset is None:
                data = np.ones((adata.n_obs,), dtype=float)
            elif isinstance(subset, List):
                data = np.asarray(adata.obs.index.isin(subset), dtype=float)
            elif isinstance(subset, tuple):
                if subset[0] >= adata.n_obs:
                    raise IndexError(f"TODO: index {subset[0]} larger than length of `adata` ({adata.n_obs}).")
                data = np.zeros((adata.n_obs,), dtype=float)
                data[range(subset[0], min(subset[0] + subset[1], adata.n_obs))] = 1.0
            else:
                raise ValueError("TODO: If `data` is `None`, `subset` needs to be `None` or a list with obs indices.")
        elif isinstance(data, str):
            if subset is None:  # allow for numeric values
                data = np.asarray(adata.obs[data], dtype=float)
            elif isinstance(subset, List) and not isinstance(subset, str):
                data = np.asarray(adata.obs[data].isin(subset), dtype=float)
            else:
                data = np.asarray(adata.obs[data].values == subset, dtype=float)
        else:
            data = np.asarray(data)

        if split_mass:
            data = _split_mass(data)

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
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self._adata

    @property
    def stage(self) -> ProblemStage:
        """Problem stage."""
        return self._stage

    @property
    def problem_kind(self) -> ProblemKind:
        """The kind of the underlying OT problem, an instance of `moscot.solvers._base_solver.ProblemKind`"""
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
        self._solver: Optional[BaseSolver[BaseSolverOutput]] = None
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
    def prepare(
        self,
        xy: Optional[Union[Mapping[str, Any], TaggedArray]] = None,
        x: Optional[Union[Mapping[str, Any], TaggedArray]] = None,
        y: Optional[Union[Mapping[str, Any], TaggedArray]] = None,
        a: Optional[Union[bool, str, ArrayLike]] = None,
        b: Optional[Union[bool, str, ArrayLike]] = None,
        **kwargs: Any,
    ) -> "OTProblem":
        """Prepare method."""
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

    @d.get_sections(base="OTProblem_solve", sections=["Parameters", "Raises"])
    @wrap_solve
    def solve(
        self,
        epsilon: Optional[float] = 1e-2,
        alpha: Optional[float] = 0.5,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        batch_size: Optional[int] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        initializer: Initializer_t = None,
        prepare_kwargs: Mapping[str, Any] = MappingProxyType({}),
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        device: Optional[Any] = None,
        **kwargs: Any,
    ) -> "OTProblem":
        """Solve method."""
        if self._problem_kind is None:
            raise RuntimeError("Run .prepare() first.")
        if self._problem_kind in (ProblemKind.QUAD, ProblemKind.QUAD_FUSED):
            kwargs["epsilon"] = epsilon
        kwargs["rank"] = rank
        kwargs["initializer"] = initializer
        a = kwargs.pop("a", self._a)
        b = kwargs.pop("b", self._b)

        prepare_kwargs = dict(prepare_kwargs)
        prepare_kwargs["epsilon"] = epsilon
        prepare_kwargs["alpha"] = alpha
        prepare_kwargs["scale_cost"] = ScaleCost(scale_cost) if isinstance(scale_cost, str) else scale_cost
        prepare_kwargs["batch_size"] = batch_size

        self._solver = self._problem_kind.solver(backend="ott", **kwargs, **initializer_kwargs)
        self._solution = self._solver(
            x=self._x,
            y=self._y,
            xy=self._xy,
            a=a,
            b=b,
            tau_a=tau_a,
            tau_b=tau_b,
            device=device,
            **prepare_kwargs,
        )
        return self

    @require_solution
    def push(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        normalize: bool = True,
        *,
        split_mass: bool = False,
        **kwargs: Any,
    ) -> ArrayLike:
        """Push mass."""
        if TYPE_CHECKING:
            assert isinstance(self.solution, BaseSolverOutput)
        data = self._get_mass(self.adata, data=data, subset=subset, normalize=normalize, split_mass=split_mass)
        return self.solution.push(data, **kwargs)

    @require_solution
    def pull(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        normalize: bool = True,
        *,
        split_mass: bool = False,
        **kwargs: Any,
    ) -> ArrayLike:
        """Pull mass."""
        adata = self.adata if self._adata_y is None else self._adata_y
        if TYPE_CHECKING:
            assert isinstance(self.solution, BaseSolverOutput)
        data = self._get_mass(adata, data=data, subset=subset, normalize=normalize, split_mass=split_mass)
        return self.solution.pull(data, **kwargs)

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

        n_comps = kwargs.pop("n_comps", 30)  # set n_comps=30 as default

        logger.info("Computing pca with `n_comps = {n_comps}` and `joint_space = {joint_space}`.")

        if return_linear:
            n = x.shape[0]
            joint_space = kwargs.pop("joint_space", True)
            if joint_space:
                data = sc.pp.pca(concat(x, y), n_comps=n_comps, **kwargs)
            else:
                data = concat(sc.pp.pca(x, n_comps=n_comps, **kwargs), sc.pp.pca(y, n_comps=n_comps, **kwargs))

            return {"xy": TaggedArray(data[:n], data[n:], tag=Tag.POINT_CLOUD)}

        x = sc.pp.pca(x, n_comps=n_comps, **kwargs)
        y = sc.pp.pca(y, n_comps=n_comps, **kwargs)
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
    def shape(self) -> Tuple[int, int]:
        return self.adata.n_obs, self._adata_y.n_obs

    @property
    def solution(self) -> Optional[BaseSolverOutput]:
        return self._solution

    @property
    def x(self) -> Optional[TaggedArray]:
        return self._x

    @property
    def y(self) -> Optional[TaggedArray]:
        return self._y

    # TODO(michalk8): verify type
    @property
    def xy(self) -> Optional[TaggedArray]:
        return self._xy

    @property
    def a(self) -> Optional[ArrayLike]:
        return self._a

    @property
    def b(self) -> Optional[ArrayLike]:
        return self._b

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[shape={self.adata.n_obs, self._adata_y.n_obs}]"

    def __str__(self) -> str:
        return repr(self)

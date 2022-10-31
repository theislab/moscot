from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Literal, Mapping, Optional, TYPE_CHECKING

from scipy.sparse import vstack, issparse, csr_matrix

import numpy as np

from anndata import AnnData
import scanpy as sc

from moscot._types import Device_t, ArrayLike
from moscot._logging import logger
from moscot._docs._docs import d
from moscot.problems._utils import wrap_solve, wrap_prepare, require_solution
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._base_solver import OTSolver, ProblemKind
from moscot._constants._constants import ProblemStage
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ["BaseProblem", "OTProblem", "ProblemKind"]


@d.get_sections(base="BaseProblem", sections=["Parameters", "Raises"])
@d.dedent
class BaseProblem(ABC):
    """Problem interface handling one optimal transport problem.

    Parameters
    ----------
    kwargs
        Metadata.
    """

    def __init__(self, **kwargs: Any):
        self._problem_kind: ProblemKind = ProblemKind.UNKNOWN
        self._stage = ProblemStage.INITIALIZED
        self._metadata = dict(kwargs)

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        """Abstract prepare method."""

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        """Abstract solve method."""

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
            elif isinstance(subset, list):
                data = np.asarray(adata.obs_names.isin(subset), dtype=float)
            elif isinstance(subset, tuple):
                # TODO(michalk8): handle negative indices
                start, offset = subset
                if start >= adata.n_obs:
                    raise IndexError(f"Expected starting index to be smaller than `{adata.n_obs}`, found `{start}`.")
                data = np.zeros((adata.n_obs,), dtype=float)
                data[range(start, min(start + offset, adata.n_obs))] = 1.0
            else:
                raise TypeError(f"Unable to interpret subset of type `{type(subset)}`.")
        elif not hasattr(data, "shape"):
            if subset is None:  # allow for numeric values
                data = np.asarray(adata.obs[data], dtype=float)
            else:
                if isinstance(subset, str):
                    subset = [subset]
                data = np.asarray(adata.obs[data].isin(subset), dtype=float)
        else:
            data = np.asarray(data, dtype=float)

        if split_mass:
            data = _split_mass(data)

        if data.ndim != 2:
            data = np.reshape(data, (-1, 1))
        if data.shape[0] != adata.n_obs:
            raise ValueError(f"Expected array of shape `({adata.n_obs,}, ...)`, found `{data.shape}`.")
        if np.any(data < 0.0):
            raise ValueError("Some entries have negative mass.")
        total = np.sum(data, axis=0, keepdims=True)
        if np.any(total <= 0.0):
            raise ValueError("Some measures have no mass.")
        return (data / total) if normalize else data

    @property
    def stage(self) -> ProblemStage:
        """Problem stage."""
        return self._stage

    @property
    def problem_kind(self) -> ProblemKind:
        """Kind of the underlying problem."""
        return self._problem_kind


@d.get_sections(base="OTProblem", sections=["Parameters", "Raises"])
@d.dedent
class OTProblem(BaseProblem):
    """
    Base class for all optimal transport problems.

    Parameters
    ----------
    adata
        Source annotated data object.
    adata_tgt
        Target annotated data object. If `None`, use ``adata``.
    src_obs_mask
        Source observation mask that defines :attr:`adata_src`.
    tgt_obs_mask
        Target observation mask that defines :attr:`adata_tgt`.
    src_var_mask
        Source variable mask that defines :attr:`adata_src`.
    tgt_var_mask
        Target variable mask that defines :attr:`adata_tgt`.
    src_key
        Source key name, usually supplied by :class:`moscot.problems.CompoundBaseProblem`.
    tgt_key
        Target key name, usually supplied by :class:`moscot.problems.CompoundBaseProblem`.
    kwargs
        Keyword arguments for :class:`moscot.problems.base.BaseProblem.`

    Notes
    -----
    If any of the source/target masks are specified, :attr:`adata_src`/:attr:`adata_tgt` will be a view.
    """

    def __init__(
        self,
        adata: AnnData,
        adata_tgt: Optional[AnnData] = None,
        src_obs_mask: Optional[ArrayLike] = None,
        tgt_obs_mask: Optional[ArrayLike] = None,
        src_var_mask: Optional[ArrayLike] = None,
        tgt_var_mask: Optional[ArrayLike] = None,
        src_key: Optional[Any] = None,
        tgt_key: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._adata_src = adata
        self._adata_tgt = adata if adata_tgt is None else adata_tgt
        self._src_obs_mask = src_obs_mask
        self._tgt_obs_mask = tgt_obs_mask
        self._src_var_mask = src_var_mask
        self._tgt_var_mask = tgt_var_mask
        self._src_key = src_key
        self._tgt_key = tgt_key

        self._solver: Optional[OTSolver[BaseSolverOutput]] = None
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
                return TaggedArray.from_adata(
                    self.adata_src, dist_key=(self._src_key, self._tgt_key), attr=attr, **kwargs
                )
            raise ValueError(f"Storing `{kwargs['tag']!r}` in `adata.{attr}` is disallowed.")

        x_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("x_")}
        y_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("y_")}
        x_kwargs["tag"] = Tag.POINT_CLOUD
        y_kwargs["tag"] = Tag.POINT_CLOUD

        x_array = TaggedArray.from_adata(self.adata_src, dist_key=self._src_key, **x_kwargs)
        y_array = TaggedArray.from_adata(self.adata_tgt, dist_key=self._tgt_key, **y_kwargs)

        # restich together
        return TaggedArray(data_src=x_array.data_src, data_tgt=y_array.data_src, tag=Tag.POINT_CLOUD, cost=x_array.cost)

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
        """Prepare optimal transport problem.

        Depending on which arguments are passed:

        - if only ``xy`` is passed, :attr:`problem_kind` will be :attr:`~moscot.solvers.ProblemKind.LINEAR`.
        - if only ``x`` and ``y`` are passed, :attr:`problem_kind` will be
          :attr:`~moscot.solvers.ProblemKind.QUAD`.
        - if all ``xy``, ``x`` and ``y`` are passed, :attr:`problem_kind` will be
          :attr:`~moscot.solvers.ProblemKind.QUAD_FUSED`.

        Parameters
        ----------
        xy
            Geometry defining the linear term. If passed as a :class:`dict`,
            :meth:`~moscot.solvers.TaggedArray.from_adata` will be called.
        x
            First geometry defining the quadratic term. If passed as a :class:`dict`,
            :meth:`~moscot.solvers.TaggedArray.from_adata` will be called.
        y
            Second geometry defining the quadratic term. If passed as a :class:`dict`,
            :meth:`~moscot.solvers.TaggedArray.from_adata` will be called.
        a
            Source marginals. Valid value are:

            - :class:`str`: key in :attr:`adata_src` :attr:`~anndata.AnnData.obs` where the marginals are stored.
            - :class:`bool`: if `True`, compute the marginals from :attr:`adata_src`, otherwise use uniform.
            - :class:`~numpy.ndarray`: array of shape ``[n,]`` containing the source marginals.
            - :obj:`None`: uniform marginals.
        b
            Target marginals. Valid options are:

            - :class:`str`: key in :attr:`adata_tgt` :attr:`~anndata.AnnData.obs` where the marginals are stored.
            - :class:`bool`: if `True`, compute the marginals from :attr:`adata_tgt`, otherwise use uniform.
            - :class:`~numpy.ndarray`: array of shape ``[m,]`` containing the target marginals.
            - :obj:`None`: uniform marginals.
        kwargs
            Keyword arguments when creating the source/target marginals.

        Returns
        -------
        Self and modifies the following attributes:

        - :attr:`xy`: geometry of shape ``[n, m]`` defining the linear term.
        - :attr:`x`: first geometry of shape ``[n, n]`` defining  the quadratic term.
        - :attr:`y`: second geometry of shape ``[m, m]`` defining the quadratic term.
        - :attr:`a`: source marginals of shape ``[n,]``.
        - :attr:`b`: target marginals of shape ``[m,]``.
        - :attr:`problem_kind`: kind of the optimal transport problem.
        - :attr:`solution`: set to :obj:`None`.
        """
        self._x = self._y = self._xy = self._solution = None
        # TODO(michalk8): in the future, have a better dispatch
        # fmt: off
        if xy is not None and x is None and y is None:
            self._problem_kind = ProblemKind.LINEAR
            self._xy = xy if isinstance(xy, TaggedArray) else self._handle_linear(**xy)
        elif x is not None and y is not None and xy is None:
            self._problem_kind = ProblemKind.QUAD
            if isinstance(x, TaggedArray):
                self._x = x
            else:
                self._x = TaggedArray.from_adata(self.adata_src, dist_key=self._src_key, **x)
            if isinstance(y, TaggedArray):
                self._y = y
            else:
                self._y = TaggedArray.from_adata(self.adata_tgt, dist_key=self._tgt_key, **y)
        elif xy is not None and x is not None and y is not None:
            self._problem_kind = ProblemKind.QUAD_FUSED
            self._xy = xy if isinstance(xy, TaggedArray) else self._handle_linear(**xy)
            if isinstance(x, TaggedArray):
                self._x = x
            else:
                self._x = TaggedArray.from_adata(self.adata_src, dist_key=self._src_key, **x)
            if isinstance(y, TaggedArray):
                self._y = y
            else:
                self._y = TaggedArray.from_adata(self.adata_tgt, dist_key=self._tgt_key, **y)
        else:
            raise ValueError("Unable to prepare the data. Either only supply `xy=...`, or `x=..., y=...`, or all.")
        # fmt: on

        self._a = self._create_marginals(self.adata_src, data=a, source=True, **kwargs)
        self._b = self._create_marginals(self.adata_tgt, data=b, source=False, **kwargs)
        return self

    @d.get_sections(base="OTProblem_solve", sections=["Parameters", "Raises"])
    @wrap_solve
    def solve(
        self,
        backend: Literal["ott"] = "ott",
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> "OTProblem":
        """Solve optimal transport problem.

        Parameters
        ----------
        backend
            Which backend to use, see :func:`moscot.backends.get_available_backends`.
        device
            Device where to transfer the solution, see :meth:`moscot.solvers.BaseSolverOutput.to`.
        kwargs
            Keyword arguments for :meth:`moscot.solvers.BaseSolver.__call__`.

        Returns
        -------
        Self and modifies the following attributes:

        - :attr:`solver`: optimal transport solver.
        - :attr:`solution`: optimal transport solution.
        """
        self._solver = self._problem_kind.solver(backend=backend, **kwargs)

        a = kwargs.pop("a", self._a)
        b = kwargs.pop("b", self._b)

        # TODO: add ScaleCost(scale_cost)

        self._solution = self._solver(  # type: ignore[misc]
            xy=self._xy,
            x=self._x,
            y=self._y,
            a=a,
            b=b,
            device=device,
            **kwargs,
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
        """Push mass through the :attr:`~moscot.solvers.BaseSolverOutput.transport_matrix`.

        Parameters
        ----------
        data
            Data to push through the transport matrix. Valid options are:

            - :class:`str`: key in :attr:`adata_src` :attr:`~anndata.AnnData.obs`.
            - :class:`~numpy.ndarray`: array of shape ``[n,]``.
            - :obj:`None`: depending on the ``subset``:

              - :class:`list`: observation names to push in :attr:`adata_src` :attr:`~anndata.AnnData.obs_names`.
              - :class:`tuple`: start and offset indices defining the mask.
              - :obj:`None`: uniform array of 1s.
        subset
            Push values contained only within the subset.
        normalize
            Whether to normalize the columns of ``data`` to sum to 1.
        split_mass
            Whether to split non-zero values in ``data`` into separate columns.

        Returns
        -------
        Array of shape ``[m, d]``.
        """
        if TYPE_CHECKING:
            assert isinstance(self.solution, BaseSolverOutput)
        data = self._get_mass(self.adata_src, data=data, subset=subset, normalize=normalize, split_mass=split_mass)
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
        """Pull mass through the :attr:`~moscot.solvers.BaseSolverOutput.transport_matrix`.

        Parameters
        ----------
        data
            Data to pull through the transport matrix. Valid options are:

            - :class:`str`: key in :attr:`adata_tgt` :attr:`~anndata.AnnData.obs`.
            - :class:`~numpy.ndarray`: array of shape ``[m,]``.
            - :obj:`None`: depending on the ``subset``:

              - :class:`list`: observation names to pull in :attr:`adata_tgt` :attr:`~anndata.AnnData.obs_names`.
              - :class:`tuple`: start and offset indices defining the mask.
              - :obj:`None`: uniform array of 1s.
        subset
            Pull values contained only within the subset.
        normalize
            Whether to normalize the columns of ``data`` to sum to 1.
        split_mass
            Whether to split non-zero values in ``data`` into separate columns.

        Returns
        -------
        Array of shape ``[n, d]``.
        """
        if TYPE_CHECKING:
            assert isinstance(self.solution, BaseSolverOutput)
        data = self._get_mass(self.adata_tgt, data=data, subset=subset, normalize=normalize, split_mass=split_mass)
        return self.solution.pull(data, **kwargs)

    @staticmethod
    def _local_pca_callback(
        adata: AnnData,
        adata_y: AnnData,
        layer: Optional[str] = None,
        return_linear: bool = True,
        n_comps: int = 30,
        **kwargs: Any,
    ) -> Dict[Literal["xy", "x", "y"], TaggedArray]:
        def concat(x: ArrayLike, y: ArrayLike) -> ArrayLike:
            if issparse(x):
                return vstack([x, csr_matrix(y)])
            if issparse(y):
                return vstack([csr_matrix(x), y])
            return np.vstack([x, y])

        if layer is None:
            x, y, msg = adata.X, adata_y.X, "adata.X"
        else:
            x, y, msg = adata.layers[layer], adata_y.layers[layer], f"adata.layers[{layer!r}]"

        if return_linear:
            n = x.shape[0]
            data = concat(x, y)
            if data.shape[1] <= n_comps:
                # TODO(michalk8): log
                return {"xy": TaggedArray(data[:n], data[n:], tag=Tag.POINT_CLOUD)}

            logger.info(f"Computing pca with `n_comps={n_comps}` using `{msg}`")
            data = sc.pp.pca(data, n_comps=n_comps, **kwargs)
            return {"xy": TaggedArray(data[:n], data[n:], tag=Tag.POINT_CLOUD)}

        logger.info(f"Computing pca with `n_comps={n_comps}` using `{msg}`")
        x = sc.pp.pca(x, n_comps=n_comps, **kwargs)
        y = sc.pp.pca(y, n_comps=n_comps, **kwargs)
        return {"x": TaggedArray(x, tag=Tag.POINT_CLOUD), "y": TaggedArray(y, tag=Tag.POINT_CLOUD)}

    def _create_marginals(
        self, adata: AnnData, *, source: bool, data: Optional[Union[bool, str, ArrayLike]] = None, **kwargs: Any
    ) -> ArrayLike:
        if data is True:
            marginals = self._estimate_marginals(adata, source=source, **kwargs)
        elif data in (False, None):
            marginals = np.ones((adata.n_obs,), dtype=float) / adata.n_obs
        elif isinstance(data, str):
            try:
                marginals = np.asarray(adata.obs[data], dtype=float)
            except KeyError:
                raise KeyError(f"Unable to find data in `adata.obs[{data!r}]`.") from None
        else:
            marginals = np.asarray(data, dtype=float)

        if marginals.shape != (adata.n_obs,):
            raise ValueError(
                f"Expected {'source' if source else 'target'} marginals "
                f"to have shape `{adata.n_obs,}`, found `{marginals.shape}`."
            )
        return marginals

    def _estimate_marginals(self, adata: AnnData, *, source: bool, **kwargs: Any) -> ArrayLike:
        return np.ones((adata.n_obs,), dtype=float) / adata.n_obs

    @property
    def adata_src(self) -> AnnData:
        """Source annotated data object."""
        adata = self._adata_src if self._src_obs_mask is None else self._adata_src[self._src_obs_mask]
        if not adata.n_obs:
            raise ValueError("No observations in the source `AnnData`.")
        adata = adata if self._src_var_mask is None else adata[:, self._src_var_mask]
        if not adata.n_vars:
            raise ValueError("No variables in the source `AnnData`.")
        return adata

    @property
    def adata_tgt(self) -> AnnData:
        """Target annotated data object."""
        adata = self._adata_tgt if self._tgt_obs_mask is None else self._adata_tgt[self._tgt_obs_mask]
        if not adata.n_obs:
            raise ValueError("No observations in the target `AnnData`.")
        adata = adata if self._tgt_var_mask is None else adata[:, self._tgt_var_mask]
        if not adata.n_vars:
            raise ValueError("No variables in the target `AnnData`.")
        return adata

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the optimal transport problem."""
        return self.adata_src.n_obs, self.adata_tgt.n_obs

    @property
    def solution(self) -> Optional[BaseSolverOutput]:
        """Solution of the optimal transport problem."""
        return self._solution

    @property
    def solver(self) -> Optional[OTSolver[BaseSolverOutput]]:
        """Optimal transport solver."""
        return self._solver

    @property
    def xy(self) -> Optional[TaggedArray]:
        """Geometry defining the linear term."""
        return self._xy

    @property
    def x(self) -> Optional[TaggedArray]:
        """First geometry defining the quadratic term."""
        return self._x

    @property
    def y(self) -> Optional[TaggedArray]:
        """Second geometry defining the quadratic term."""
        return self._y

    @property
    def a(self) -> Optional[ArrayLike]:
        """Source marginals."""
        return self._a

    @property
    def b(self) -> Optional[ArrayLike]:
        """Target marginals."""
        return self._b

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[stage={self.stage!r}, shape={self.shape!r}]"

    def __str__(self) -> str:
        return repr(self)

import abc
import pathlib
import types
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import cloudpickle
from docstring_inheritance import NumpyDocstringInheritanceMeta

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandas.api import types as pd_types
from sklearn.preprocessing import StandardScaler

import anndata as ad
import scanpy as sc
from anndata import AnnData

from moscot import backends
from moscot._logging import logger
from moscot._types import ArrayLike, CostFn_t, Device_t, ProblemKind_t
from moscot.base.output import BaseDiscreteSolverOutput, MatrixSolverOutput
from moscot.base.problems._utils import (
    TimeScalesHeatKernel,
    _assert_columns_and_index_match,
    _assert_series_match,
    require_solution,
    wrap_prepare,
    wrap_solve,
)
from moscot.base.solver import OTSolver
from moscot.utils.subset_policy import (  # type:ignore[attr-defined]
    ExplicitPolicy,
    Policy_t,
    StarPolicy,
    SubsetPolicy,
    create_policy,
)
from moscot.utils.tagged_array import (
    DistributionCollection,
    DistributionContainer,
    Tag,
    TaggedArray,
)

K = TypeVar("K", bound=Hashable)

__all__ = ["BaseProblem", "OTProblem", "CondOTProblem"]


class CombinedMeta(abc.ABCMeta, NumpyDocstringInheritanceMeta):
    pass


class BaseProblem(abc.ABC, metaclass=CombinedMeta):
    """Base class for all :term:`OT` problems."""

    def __init__(self):
        self._problem_kind: ProblemKind_t = "unknown"
        self._stage: Literal["initialized", "prepared", "solved"] = "initialized"

    @abc.abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        """Prepare the problem.

        Parameters
        ----------
        args
            Positional arguments.
        kwargs
            Keyword arguments.

        Returns
        -------
        Self and updates the following fields:

        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - kind of the :term:`OT` problem.
        """

    @abc.abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> "BaseProblem":
        """Solve the problem.

        Parameters
        ----------
        args
            Positional arguments.
        kwargs
            Keyword arguments.

        Returns
        -------
        Self and updates the following fields:

        - :attr:`stage` - set to ``'solved'``.
        - :attr:`problem_kind` - kind of the :term:`OT` problem.
        """

    def save(
        self,
        path: Union[str, pathlib.Path],
        overwrite: bool = False,
    ) -> None:
        """Save the problem to a file.

        Parameters
        ----------
        path
            Path where to save the problem.
        overwrite
            Whether to overwrite an existing file.

        Returns
        -------
        Nothing, just saves the problem using :mod:`cloudpickle <pickle>`.
        """
        path = pathlib.Path(path)
        if not overwrite and path.is_file():
            raise RuntimeError(
                f"Unable to write the model to an existing file `{path}`, " f"use `overwrite=True` to overwrite it."
            )
        with open(path, "wb") as fout:
            cloudpickle.dump(self, fout)

    @staticmethod
    def load(
        path: Union[str, pathlib.Path],
    ) -> "BaseProblem":
        """Load the model from a file.

        Parameters
        ----------
        path
            Path where the model is stored.

        Returns
        -------
        The problem.
        """
        with open(path, "rb") as fin:
            return cloudpickle.load(fin)

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
                data = (
                    np.asarray(adata.obs[data], dtype=float)
                    if pd_types.is_numeric_dtype(adata.obs[data])
                    else np.ones((adata.n_obs,), dtype=float)
                )
            else:
                sset = subset if isinstance(subset, list) else [subset]  # type:ignore[list-item]
                data = np.asarray(adata.obs[data].isin(sset), dtype=float)
        else:
            data = np.asarray(data, dtype=float)

        if split_mass:
            data = _split_mass(data)

        if data.ndim != 2:
            data = np.reshape(data, (-1, 1))
        if data.shape[0] != adata.n_obs:
            raise ValueError(f"Expected array of shape `({adata.n_obs,}, ...)`, found `{data.shape}`.")

        if normalize:
            return data / np.sum(data, axis=0, keepdims=True)
        return data

    @property
    def stage(self) -> Literal["initialized", "prepared", "solved"]:
        """Problem stage."""
        return self._stage

    @property
    def problem_kind(self) -> ProblemKind_t:
        """Kind of the underlying problem."""
        return self._problem_kind


class AbstractAdataAccess(ABC, metaclass=CombinedMeta):

    @property
    @abstractmethod
    def adata(self) -> AnnData:
        """Annotated data object."""
        pass


class AbstractPushPullAdata(AbstractAdataAccess):

    @abstractmethod
    def pull(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def push(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def _apply(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass


class AbstractSolutionsProblems(ABC, metaclass=CombinedMeta):

    @property
    @abstractmethod
    def solutions(self) -> Any:
        """Solutions."""
        pass

    @property
    @abstractmethod
    def problems(self) -> Any:
        """Problems."""
        pass

    @property
    @abstractmethod
    def _policy(self) -> Any:
        """Subset policy."""
        pass


class AbstractSrcTgt(ABC, metaclass=CombinedMeta):

    @property
    @abstractmethod
    def adata_src(self) -> AnnData:
        """Annotated data object."""
        pass

    @property
    @abstractmethod
    def adata_tgt(self) -> AnnData:
        """Annotated data object."""
        pass


class AbstractSpSc(ABC, metaclass=CombinedMeta):

    @property
    @abstractmethod
    def adata_sp(self) -> AnnData:
        """Annotated data object."""
        pass

    @property
    @abstractmethod
    def adata_sc(self) -> AnnData:
        """Annotated data object."""
        pass


class OTProblem(BaseProblem):
    """Base class for all :term:`OT` problems.

    Parameters
    ----------
    adata
        Source annotated data object.
    adata_tgt
        Target annotated data object. If :obj:`None`, use ``adata``.
    src_obs_mask
        Source observation mask that defines :attr:`adata_src`.
    tgt_obs_mask
        Target observation mask that defines :attr:`adata_tgt`.
    src_var_mask
        Source variable mask that defines :attr:`adata_src`.
    tgt_var_mask
        Target variable mask that defines :attr:`adata_tgt`.
    src_key
        Source key name, usually supplied by the :class:`~moscot.base.problems.BaseCompoundProblem`.
    tgt_key
        Target key name, usually supplied by the :class:`~moscot.base.problems.BaseCompoundProblem`.

    Notes
    -----
    If source/target mask is specified, :attr:`adata_src`/:attr:`adata_tgt` will be a view.
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
    ):
        super().__init__()
        self._adata_src = adata
        self._adata_tgt = adata if adata_tgt is None else adata_tgt
        self._src_obs_mask = src_obs_mask
        self._tgt_obs_mask = tgt_obs_mask
        self._src_var_mask = src_var_mask
        self._tgt_var_mask = tgt_var_mask
        self._src_key = src_key
        self._tgt_key = tgt_key

        self._solver: Optional[OTSolver[BaseDiscreteSolverOutput]] = None
        self._solution: Optional[BaseDiscreteSolverOutput] = None

        self._x: Optional[TaggedArray] = None
        self._y: Optional[TaggedArray] = None
        self._xy: Optional[TaggedArray] = None

        self._a: Optional[ArrayLike] = None
        self._b: Optional[ArrayLike] = None

        self._time_scales_heat_kernel = TimeScalesHeatKernel(None, None, None)

    def _handle_linear(self, cost: CostFn_t = None, **kwargs: Any) -> TaggedArray:
        if "x_attr" not in kwargs or "y_attr" not in kwargs:
            kwargs.setdefault("tag", Tag.COST_MATRIX)
            attr = kwargs.pop("attr", "obsm")
            if attr in ("obsm", "uns"):
                return TaggedArray.from_adata(
                    self.adata_src, dist_key=(self._src_key, self._tgt_key), attr=attr, cost="custom", **kwargs
                )
            raise ValueError(f"Storing `{kwargs['tag']!r}` in `adata.{attr}` is disallowed.")

        x_kwargs, y_kwargs = self._split_xy_kwargs(**kwargs)
        if cost is not None:
            x_kwargs["cost"] = cost
            y_kwargs["cost"] = cost
        x_kwargs["tag"] = Tag.POINT_CLOUD
        y_kwargs["tag"] = Tag.POINT_CLOUD

        x_array = TaggedArray.from_adata(self.adata_src, dist_key=self._src_key, **x_kwargs)
        y_array = TaggedArray.from_adata(self.adata_tgt, dist_key=self._tgt_key, **y_kwargs)

        return TaggedArray(data_src=x_array.data_src, data_tgt=y_array.data_src, tag=Tag.POINT_CLOUD, cost=x_array.cost)

    @wrap_prepare
    def prepare(
        self,
        xy: Mapping[str, Any],
        x: Mapping[str, Any],
        y: Mapping[str, Any],
        a: Optional[Union[bool, str, ArrayLike]] = None,
        b: Optional[Union[bool, str, ArrayLike]] = None,
        marginal_kwargs: Dict[str, Any] = types.MappingProxyType({}),
    ) -> "OTProblem":
        """Prepare the :term:`OT` problem.

        Depending on which arguments are passed, the :attr:`problem_kind` will be:

        - if only ``xy`` is non-empty, :attr:`problem_kind = 'linear' <problem_kind>`.
        - if only ``x`` and ``y`` are non-empty, :attr:`problem_kind = 'quadratic' <problem_kind>`.
        - if all ``xy``, ``x`` and ``y`` are non-empty, :attr:`problem_kind = 'quadratic' <problem_kind>`.

        Parameters
        ----------
        xy
            Geometry defining the linear term. If a non-empty :class:`dict`,
            :meth:`~moscot.utils.tagged_array.TaggedArray.from_adata` will be called.
        x
            Geometry defining the source :term:`quadratic term`. If a non-empty :class:`dict`,
            :meth:`~moscot.utils.tagged_array.TaggedArray.from_adata` will be called.
        y
            Geometry defining the target :term:`quadratic term`. If a non-empty :class:`dict`,
            :meth:`~moscot.utils.tagged_array.TaggedArray.from_adata` will be called.
        a
            Source :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the source marginals are stored.
            - :class:`bool` - if :obj:`True`, :meth:`estimate the marginals <estimate_marginals>`
              from :attr:`adata_src`, otherwise use uniform marginals.
            - :class:`~numpy.ndarray` - array of shape ``[n,]`` containing the source marginals.
            - :obj:`None` - uniform marginals.
        b
            Target :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the target marginals are stored.
            - :class:`bool` - if :obj:`True`, :meth:`estimate the marginals <estimate_marginals>`
              from :attr:`adata_tgt`, otherwise use uniform marginals.
            - :class:`~numpy.ndarray` - array of shape ``[m,]`` containing the target marginals.
            - :obj:`None` - uniform marginals.
        marginal_kwargs
            Keyword arguments for :meth:`estimate_marginals` when ``a = True`` or ``b = True``.

        Returns
        -------
        Self and updates the following fields:

        - :attr:`xy` - geometry of shape ``[n, m]`` defining the :term:`linear term`.
        - :attr:`x` - geometry of shape ``[n, n]`` defining the source :term:`quadratic term`.
        - :attr:`y` - geometry of shape ``[m, m]`` defining the target :term:`quadratic term`.
        - :attr:`a` -  source :term:`marginals` of shape ``[n,]``.
        - :attr:`b` - target :term:`marginals` of shape ``[m,]``.
        - :attr:`solution` - set to :obj:`None`.
        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - kind of the :term:`OT` problem.
        """
        self._x = self._y = self._xy = self._solution = None
        # TODO(michalk8): in the future, have a better dispatch
        # fmt: off
        if xy:
            if "tagged_array" in xy:
                kws, _ = self._split_xy_kwargs(**xy)
                self._xy = xy["tagged_array"]._set_cost(**kws)
            else:
                self._xy = self._handle_linear(**xy)
        if x:
            if "tagged_array" in x:
                x = dict(x)
                self._x = x.pop("tagged_array")._set_cost(**x)
            else:
                self._x = TaggedArray.from_adata(self.adata_src, dist_key=self._src_key, **x)
        if y:
            if "tagged_array" in y:
                y = dict(y)
                self._y = y.pop("tagged_array")._set_cost(**y)
            else:
                self._y = TaggedArray.from_adata(self.adata_tgt, dist_key=self._tgt_key, **y)
        if self._xy and not self._x and not self._y:
            self._problem_kind = "linear"
        elif (self._x and self._y and not self._xy) or (self._x and self._y and self._xy):
            self._problem_kind = "quadratic"
        else:
            raise ValueError("Unable to prepare the data. Either only supply `xy=...`, or `x=..., y=...`, or all.")
        # fmt: on
        self._a = self._create_marginals(self.adata_src, data=a, source=True, marginal_kwargs=marginal_kwargs)
        self._b = self._create_marginals(self.adata_tgt, data=b, source=False, marginal_kwargs=marginal_kwargs)
        return self

    @wrap_solve
    def solve(
        self,
        backend: Literal["ott"] = "ott",
        solver_name: Optional[str] = None,
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> "OTProblem":
        """Solve the :term:`OT` problem.

        Parameters
        ----------
        backend
            Which backend to use, see :func:`~moscot.backends.utils.get_available_backends`.
        solver_name
            Literal defining the solver. If `None`, automatically infers the discrete OT solver.
        device
            Transfer the solution to a different device, see :meth:`~moscot.base.output.BaseDiscreteSolverOutput.to`.
            If :obj:`None`, keep the output on the original device.
        kwargs
            Keyword arguments for :class:`~moscot.base.solver.BaseSolver` or its
            :meth:`__call__ <moscot.base.solver.BaseSolver.__call__>` method.

        Returns
        -------
        Self and updates the following fields:

        - :attr:`solver` - the :term:`OT` solver.
        - :attr:`solution` - the :term:`OT` solution.
        """
        solver_class = backends.get_solver(
            self.problem_kind, solver_name=solver_name, backend=backend, return_class=True
        )

        init_kwargs, call_kwargs = solver_class._partition_kwargs(**kwargs)
        self._solver = solver_class(**init_kwargs)

        # note that the solver call consists of solver._prepare and solver._solve
        self._solution = self._solver(  # type: ignore[misc]
            xy=self._xy,
            x=self._x,
            y=self._y,
            a=self.a,
            b=self.b,
            device=device,
            time_scales_heat_kernel=self._time_scales_heat_kernel,
            **call_kwargs,
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
        scale_by_marginals: bool = False,
    ) -> ArrayLike:
        r"""Push data through the :attr:`~moscot.base.output.BaseDiscreteSolverOutput.transport_matrix`.

        Parameters
        ----------
        data
            Data to push through the transport matrix. Valid options are:

            - :class:`~numpy.ndarray` - array of shape ``[n,]`` or ``[n, d]``.
            - :class:`str` - key in :attr:`adata_src.obs['{data}'] <adata_src>`. If ``subset`` is a :class:`list`,
              the data will be a boolean mask determined by the subset. Useful for categorical data.
            - :obj:`None` - the value depends on the ``subset``.

              - :class:`list` - names in :attr:`adata_src.obs_names <adata_src>` to push.
              - :class:`tuple` - start and offset indices :math:`(subset[0], subset[0] + subset[1])`.
                that define a boolean mask to push.
              - :obj:`None` - uniform array of :math:`1`.
        subset
            Push values contained only within the subset.
        normalize
            Whether to normalize the columns of ``data`` to sum to :math:`1`.
        split_mass
            Whether to split non-zero values in ``data`` into separate columns.
        scale_by_marginals
            Whether to scale by the source :term`marginals` :attr:`a`.

        Returns
        -------
        The transported values, array of shape ``[m, d]``.
        """
        if TYPE_CHECKING:
            assert isinstance(self.solution, BaseDiscreteSolverOutput)
        data = self._get_mass(self.adata_src, data=data, subset=subset, normalize=normalize, split_mass=split_mass)
        return self.solution.push(data, scale_by_marginals=scale_by_marginals)

    @require_solution
    def pull(
        self,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        normalize: bool = True,
        *,
        split_mass: bool = False,
        scale_by_marginals: bool = False,
    ) -> ArrayLike:
        r"""Pull data through the :attr:`~moscot.base.output.BaseDiscreteSolverOutput.transport_matrix`.

        Parameters
        ----------
        data
            Data to pull through the transport matrix. Valid options are:

            - :class:`~numpy.ndarray` - array of shape ``[m,]`` or ``[m, d]``.
            - :class:`str` - key in :attr:`adata_tgt.obs['{data}'] <adata_tgt>`. If ``subset`` is a :class:`list`,
              the data will be a boolean mask determined by the subset. Useful for categorical data.
            - :obj:`None` - the value depends on the ``subset``.

              - :class:`list` - names in :attr:`adata_tgt.obs_names <adata_tgt>` to pull.
              - :class:`tuple` - start and offset indices :math:`(subset[0], subset[0] + subset[1])`.
                that define a boolean mask to pull.
              - :obj:`None` - uniform array of :math:`1`.
        subset
            Pull values contained only within the subset.
        normalize
            Whether to normalize the columns of ``data`` to sum to :math:`1`.
        split_mass
            Whether to split non-zero values in ``data`` into separate columns.
        scale_by_marginals
            Whether to scale by the target :term`marginals` :attr:`b`.

        Returns
        -------
        The transported values, array of shape ``[n, d]``.
        """
        if TYPE_CHECKING:
            assert isinstance(self.solution, BaseDiscreteSolverOutput)
        data = self._get_mass(self.adata_tgt, data=data, subset=subset, normalize=normalize, split_mass=split_mass)
        return self.solution.pull(data, scale_by_marginals=scale_by_marginals)

    def set_solution(
        self,
        solution: Union[ArrayLike, pd.DataFrame, BaseDiscreteSolverOutput],
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> "OTProblem":
        """Set a :attr:`solution` to the :term:`OT` problem.

        Parameters
        ----------
        solution
            Solution for this problem. If a :class:`~pandas.DataFrame` is passed,
            its index must be equal to :attr:`adata_src.obs_names <adata_src>`
            and its columns to :attr:`adata_tgt.obs_names <adata_tgt>`.
        overwrite
            Whether to overwrite an existing solution.
        kwargs
            Keyword arguments for :class:`~moscot.base.output.MatrixSolverOutput`.

        Returns
        -------
        Self and updates the following fields:

        - :attr:`solution` - the :term:`OT` solution.
        - :attr:`stage` - set to ``'solved'``.
        """
        if not overwrite and self.solution is not None:
            raise ValueError(f"`{self}` already contains a solution, use `overwrite=True` to overwrite it.")

        if isinstance(solution, pd.DataFrame):
            _assert_series_match(self.adata_src.obs_names.to_series(), solution.index.to_series())
            _assert_series_match(self.adata_tgt.obs_names.to_series(), solution.columns.to_series())
            solution = solution.to_numpy()
        if not isinstance(solution, BaseDiscreteSolverOutput):
            solution = MatrixSolverOutput(solution, **kwargs)

        if solution.shape != self.shape:
            raise ValueError(f"Expected solution to have shape `{self.shape}`, found `{solution.shape}`.")

        self._stage = "solved"
        self._solution = solution
        return self

    @staticmethod
    def _local_pca_callback(
        term: Literal["x", "y", "xy"],
        adata: AnnData,
        adata_y: Optional[AnnData] = None,
        layer: Optional[str] = None,
        n_comps: int = 30,
        scale: bool = False,
        **kwargs: Any,
    ) -> TaggedArray:
        def concat(x: ArrayLike, y: ArrayLike) -> ArrayLike:
            if sp.issparse(x):
                return sp.vstack([x, sp.csr_matrix(y)])
            if sp.issparse(y):
                return sp.vstack([sp.csr_matrix(x), y])
            return np.vstack([x, y])

        if layer is None:
            x, y, msg = adata.X, adata_y.X if adata_y is not None else None, "adata.X"
        else:
            x, y, msg = (
                adata.layers[layer],
                adata_y.layers[layer] if adata_y is not None else None,
                f"adata.layers[{layer!r}]",
            )

        scaler = StandardScaler() if scale else None

        if term == "xy":
            if y is None:
                raise ValueError("When `term` is `xy` `adata_y` cannot be `None`.")
            n = x.shape[0]
            data = concat(x, y)
            if data.shape[1] <= n_comps:
                # TODO(michalk8): log
                return TaggedArray(data[:n], data[n:], tag=Tag.POINT_CLOUD)

            logger.info(f"Computing pca with `n_comps={n_comps}` for `xy` using `{msg}`")
            data = sc.pp.pca(data, n_comps=n_comps, **kwargs)
            if scaler is not None:
                data = scaler.fit_transform(data)
            return TaggedArray(data[:n], data[n:], tag=Tag.POINT_CLOUD)
        if term in ("x", "y"):  # if we don't have a shared space, then adata_y is always None
            logger.info(f"Computing pca with `n_comps={n_comps}` for `{term}` using `{msg}`")
            x = sc.pp.pca(x, n_comps=n_comps, **kwargs)
            if scaler is not None:
                x = scaler.fit_transform(x)
            return TaggedArray(x, tag=Tag.POINT_CLOUD)
        raise ValueError(f"Expected `term` to be one of `x`, `y`, or `xy`, found `{term!r}`.")

    # TODO(@giovp): refactor
    @staticmethod
    def _spatial_norm_callback(
        term: Literal["x", "y"],
        adata: AnnData,
        adata_y: Optional[AnnData] = None,
        attr: Optional[Literal["X", "obsp", "obsm", "layers", "uns"]] = None,
        key: Optional[str] = None,
    ) -> TaggedArray:
        if term == "y":
            if adata_y is None:
                raise ValueError("When `term` is `y`, `adata_y` cannot be `None`.")
            adata = adata_y
        if attr is None:
            raise ValueError("`attr` cannot be `None` with this callback.")
        spatial = TaggedArray._extract_data(adata, attr=attr, key=key)

        logger.info(f"Normalizing spatial coordinates of `{term}`.")
        spatial = (spatial - spatial.mean()) / spatial.std()
        return TaggedArray(spatial, tag=Tag.POINT_CLOUD)

    @staticmethod
    def _graph_construction_callback(
        term: Literal["xy", "x", "y"],
        adata: AnnData,
        adata_y: Optional[AnnData] = None,
        use_rep: str = "X_pca",
        **kwargs: Any,
    ) -> TaggedArray:
        if term == "xy":
            if adata_y is None:
                raise ValueError("When `term` is `xy`, `adata_y` cannot be `None`.")
            if use_rep not in adata.obsm:
                raise ValueError(f"Unable to find `{use_rep}` in `adata.obsm`.")
            if use_rep not in adata_y.obsm:
                raise ValueError(f"Unable to find `{use_rep}` in `adata_y.obsm`.")
            adata_concat = ad.concat((adata, adata_y), join="inner")
            logger.info(f"Computing graph construction for `xy` using `{use_rep}`")
            sc.pp.neighbors(adata_concat, use_rep=use_rep, **kwargs)
            return TaggedArray(
                data_src=adata_concat.obsp["connectivities"].astype("float64"), data_tgt=None, tag=Tag.GRAPH
            )  # TODO(@michalk8): find a better solution.

        if use_rep not in adata.obsm:
            raise ValueError(f"Unable to find `{use_rep}` in `adata.obsm`.")

        logger.info(f"Computing graph construction for `{term}` using `{use_rep}`")
        sc.pp.neighbors(adata, use_rep=use_rep, **kwargs)
        return TaggedArray(data_src=adata.obsp["connectivities"].astype("float64"), data_tgt=None, tag=Tag.GRAPH)

    def _create_marginals(
        self,
        adata: AnnData,
        *,
        source: bool,
        data: Optional[Union[bool, str, ArrayLike]] = None,
        marginal_kwargs: Dict[str, Any] = types.MappingProxyType({}),
    ) -> ArrayLike:
        if data is True:  # this is the only case when kwargs are passed
            marginals = self.estimate_marginals(adata, source=source, **marginal_kwargs)
        elif data is False or data is None:
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

    def estimate_marginals(self, adata: AnnData, *, source: bool) -> ArrayLike:
        """Estimate the source or target :term:`marginals`.

        .. note::
            This function returns uniform marginals.

        Parameters
        ----------
        adata
            Annotated data object.
        source
            Whether to estimate the source or target :term:`marginals`.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        The estimated source or target marginals of shape ``[n,]`` or ``[m,]``, depending on the ``source``.
        """
        return np.ones((adata.n_obs,), dtype=float) / adata.n_obs

    def set_graph_xy(
        self,
        data: Union[pd.DataFrame, Tuple[sp.csr_matrix, pd.Series, pd.Series]],
        cost: Literal["geodesic"] = "geodesic",
        t: Optional[float] = None,
    ) -> None:
        r"""Set a graph for the :term:`linear term` for graph based distances.

        Parameters
        ----------
        data
            Data containing the graph.

                - If of type :class:`~pandas.DataFrame`, its index must be equal to
                  :attr:`adata_src.obs_names <adata_src>` and its columns to :attr:`adata_tgt.obs_names <adata_tgt>`.
                - If of type :class:`tuple`, it must be of the form (sp.csr_matrix, pd.Series, pd.Series),
                  where the first element is the graph, the second element and the third element
                  are the annotations of the graph.

        cost
            Which graph-based distance to use.
        t
            Time parameter at which to solve the heat equation, see :cite:`crane:13`. When ``t`` is :obj:`None`,
            ``t`` will be set to :math:`\epsilon / 4`, where :math:`\epsilon` is the entropy regularization term.
            This approaches the geodesic distance and allows for linear memory complexity as the cost matrix does
            not have to be instantiated :cite:`huguet:23`.

        Returns
        -------
        Nothing, just updates the following fields:

        - :attr:`xy` - the :term:`linear term`.
        - :attr:`stage` - set to ``'prepared'``.
        """
        expected_series = pd.concat([self.adata_src.obs_names.to_series(), self.adata_tgt.obs_names.to_series()])
        if isinstance(data, pd.DataFrame):
            _assert_columns_and_index_match(expected_series, data)
            data_src = data.to_numpy()
        elif isinstance(data, tuple):
            data_src, index_src, index_tgt = data
            _assert_series_match(expected_series, index_src)
            _assert_series_match(expected_series, index_tgt)
        else:
            raise ValueError(
                "Expected data to be a `pd.DataFrame` or a tuple of (`sp.csr_matrix`, `pd.Series`, `pd.Series`), "
                f"found `{type(data)}`."
            )
        self._xy = TaggedArray(data_src=data_src, data_tgt=None, tag=Tag.GRAPH, cost=cost)
        self._stage = "prepared"
        self._time_scales_heat_kernel = self._time_scales_heat_kernel._replace(xy=t)

    def set_graph_x(
        self,
        data: Union[pd.DataFrame, Tuple[sp.csr_matrix, pd.Series]],
        cost: Literal["geodesic"] = "geodesic",
        t: Optional[float] = None,
    ) -> None:
        r"""Set a graph for the source :term:`quadratic term`.

        Parameters
        ----------
        data
            Data containing the graph.

                - If of type :class:`~pandas.DataFrame`, its index and columns must be equal to
                  :attr:`adata_src.obs_names <adata_src>`.
                - If of type :class:`tuple`, it must be of the form (sp.csr_matrix, pd.Series), where the first
                  element is the graph and the second element is the annotation of the graph.

        cost
            Which graph-based distance to use.
        t
            Time parameter at which to solve the heat equation, see :cite:`crane:13`. When ``t`` is :obj:`None`,
            ``t`` will be set to :math:`\epsilon / 4`, where :math:`\epsilon` is the entropy regularization term.
            This approaches the geodesic distance and allows for linear memory complexity as the cost matrix does
            not have to be instantiated :cite:`huguet:23`.

        Returns
        -------
        Nothing, just updates the following fields:

        - :attr:`x` - the source :term:`quadratic term`.
        - :attr:`stage` - set to ``'prepared'``.
        """
        expected_series = self.adata_src.obs_names.to_series()
        if isinstance(data, pd.DataFrame):
            _assert_columns_and_index_match(expected_series, data)
            data_src = data.to_numpy()
        elif isinstance(data, tuple):
            data_src, index_src = data
            _assert_series_match(expected_series, index_src)
        else:
            raise ValueError(
                "Expected data to be a `pd.DataFrame` or a tuple of (`sp.csr_matrix`, `pd.Series`), "
                f"found `{type(data)}`."
            )
        self._x = TaggedArray(data_src=data_src, data_tgt=None, tag=Tag.GRAPH, cost=cost)
        self._stage = "prepared"
        self._time_scales_heat_kernel = self._time_scales_heat_kernel._replace(x=t)

    def set_graph_y(
        self,
        data: Union[pd.DataFrame, Tuple[sp.csr_matrix, pd.Series]],
        cost: Literal["geodesic"] = "geodesic",
        t: Optional[float] = None,
    ) -> None:
        r"""Set a graph for the target :term:`quadratic term`.

        Parameters
        ----------
        data
            Data containing the graph.

                - If of type :class:`~pandas.DataFrame`, its index and columns must be equal to
                  :attr:`adata_tgt.obs_names <adata_tgt>`.
                - If of type :class:`tuple`, it must be of the form (sp.csr_matrix, pd.Series), where the first
                  element is the graph and the second element is the annotation of the graph.

        cost
            Which graph-based distance to use.
        t
            Time parameter at which to solve the heat equation, see :cite:`crane:13`. When ``t`` is :obj:`None`,
            ``t`` will be set to :math:`\epsilon / 4`, where :math:`\epsilon` is the entropy regularization term.
            This approaches the geodesic distance and allows for linear memory complexity as the cost matrix does
            not have to be instantiated :cite:`huguet:23`.

        Returns
        -------
        Nothing, just updates the following fields:

        - :attr:`x` - the target :term:`quadratic term`.
        - :attr:`stage` - set to ``'prepared'``.
        """
        expected_series = self.adata_tgt.obs_names.to_series()
        if isinstance(data, pd.DataFrame):
            _assert_columns_and_index_match(expected_series, data)
            data_src = data.to_numpy()
        elif isinstance(data, tuple):
            data_src, index_src = data
            _assert_series_match(expected_series, index_src)
        else:
            raise ValueError(
                "Expected data to be a `pd.DataFrame` or a tuple of (`sp.csr_matrix`, `pd.Series`), "
                f"found `{type(data)}`."
            )
        self._y = TaggedArray(data_src=data_src, data_tgt=None, tag=Tag.GRAPH, cost=cost)
        self._stage = "prepared"
        self._time_scales_heat_kernel = self._time_scales_heat_kernel._replace(y=t)

    # TODO(michalk8): extend for point-clouds as Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
    # TODO(michalk8): allow this to be nullified
    def set_xy(
        self,
        data: pd.DataFrame,
        tag: Literal["cost_matrix", "kernel"],
    ) -> None:
        """Set a cost/kernel matrix for the :term:`linear term`.

        Parameters
        ----------
        data
            Cost or kernel matrix. Its index must be equal to :attr:`adata_src.obs_names <adata_src>`
            and its columns to :attr:`adata_tgt.obs_names <adata_tgt>`.
        tag
            Whether ``data`` is a cost or a kernel matrix.

        Returns
        -------
        Nothing, just updates the following fields:

        - :attr:`xy` - the :term:`linear term`.
        - :attr:`stage` - set to ``'prepared'``.
        """
        _assert_series_match(self.adata_src.obs_names.to_series(), data.index.to_series())
        _assert_series_match(self.adata_tgt.obs_names.to_series(), data.columns.to_series())

        self._xy = TaggedArray(data_src=data.to_numpy(), data_tgt=None, tag=Tag(tag), cost="cost")
        self._stage = "prepared"

    def set_x(self, data: pd.DataFrame, tag: Literal["cost_matrix", "kernel"]) -> None:
        """Set a cost/kernel matrix for the source :term:`quadratic term`.

        Parameters
        ----------
        data
            Cost or kernel matrix. Its index must be equal to :attr:`adata_src.obs_names <adata_src>`
            and its columns to :attr:`adata_src.obs_names <adata_src>`.
        tag
            Whether ``data`` is a cost or a kernel matrix.

        Returns
        -------
        Nothing, just updates the following fields:

        - :attr:`x` - the source :term:`quadratic term`.
        - :attr:`stage` - set to ``'prepared'``.
        """
        _assert_columns_and_index_match(self.adata_src.obs_names.to_series(), data)

        if self.problem_kind == "linear":
            logger.info(f"Changing the problem type from {self.problem_kind!r} to 'quadratic (fused)'.")
            self._problem_kind = "quadratic"
        self._x = TaggedArray(data_src=data.to_numpy(), data_tgt=None, tag=Tag(tag), cost="cost")
        self._stage = "prepared"

    def set_y(self, data: pd.DataFrame, tag: Literal["cost_matrix", "kernel"]) -> None:
        """Set a cost/kernel matrix for the target :term:`quadratic term`.

        Parameters
        ----------
        data
            Cost or kernel matrix. Its index must be equal to :attr:`adata_tgt.obs_names <adata_tgt>`
            and its columns to :attr:`adata_tgt.obs_names <adata_tgt>`.
        tag
            Whether ``data`` is a cost or a kernel matrix.

        Returns
        -------
        Nothing, just updates the following fields:

        - :attr:`y` - the target :term:`quadratic term`.
        - :attr:`stage` - set to ``'prepared'``.
        """
        _assert_columns_and_index_match(self.adata_tgt.obs_names.to_series(), data)

        if self.problem_kind == "linear":
            logger.info(f"Changing the problem type from {self.problem_kind!r} to 'quadratic (fused)'.")
            self._problem_kind = "quadratic"
        self._y = TaggedArray(data_src=data.to_numpy(), data_tgt=None, tag=Tag(tag), cost="cost")
        self._stage = "prepared"

    @staticmethod
    def _split_xy_kwargs(**kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        x_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("x_")}
        y_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("y_")}
        return x_kwargs, y_kwargs

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
        """Shape of the :term:`OT` problem."""
        return self.adata_src.n_obs, self.adata_tgt.n_obs

    @property
    def solution(self) -> Optional[BaseDiscreteSolverOutput]:
        """Solution of the :term:`OT` problem."""
        return self._solution

    @property
    def solver(self) -> Optional[OTSolver[BaseDiscreteSolverOutput]]:
        """:term:`OT` solver."""
        return self._solver

    @property
    def xy(self) -> Optional[TaggedArray]:
        """Geometry defining the :term:`linear term`."""
        return self._xy

    @property
    def x(self) -> Optional[TaggedArray]:
        """Geometry defining the source :term:`quadratic term`."""
        return self._x

    @property
    def y(self) -> Optional[TaggedArray]:
        """Geometry defining the target :term:`quadratic term`."""
        return self._y

    @property
    def a(self) -> Optional[ArrayLike]:
        """Source :term:`marginals`."""
        return self._a

    @property
    def b(self) -> Optional[ArrayLike]:
        """Target :term:`marginals`."""
        return self._b

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[stage={self.stage!r}, shape={self.shape!r}]"

    def __str__(self) -> str:
        return repr(self)


class CondOTProblem(BaseProblem):  # TODO(@MUCDK) check generic types, save and load
    """
    Base class for all conditional (nerual) optimal transport problems.

    Parameters
    ----------
    adata
        Source annotated data object.
    kwargs
        Keyword arguments for :class:`moscot.base.problems.problem.BaseProblem`
    """

    def __init__(
        self,
        adata: AnnData,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._adata = adata

        self._distributions: Optional[DistributionCollection[K]] = None  # type: ignore[valid-type]
        self._policy: Optional[SubsetPolicy[Any]] = None
        self._sample_pairs: Optional[List[Tuple[Any, Any]]] = None

        self._solver: Optional[OTSolver[BaseDiscreteSolverOutput]] = None
        self._solution: Optional[BaseDiscreteSolverOutput] = None

        self._a: Optional[str] = None
        self._b: Optional[str] = None

    @wrap_prepare
    def prepare(
        self,
        policy_key: str,
        policy: Policy_t,
        xy: Mapping[str, Any],
        xx: Mapping[str, Any],
        conditions: Mapping[str, Any],
        a: Optional[str] = None,
        b: Optional[str] = None,
        subset: Optional[Sequence[Tuple[K, K]]] = None,
        reference: K = None,
        **kwargs: Any,
    ) -> "CondOTProblem":
        """Prepare conditional optimal transport problem.

        Parameters
        ----------
        xy
            Geometry defining the linear term. If passed as a :class:`dict`,
            :meth:`~moscot.utils.tagged_array.TaggedArray.from_adata` will be called.
        policy
            Policy defining which pairs of distributions to sample from during training.
        policy_key
            %(key)s
        a
            Source marginals.
        b
            Target marginals.
        kwargs
            Keyword arguments when creating the source/target marginals.


        Returns
        -------
        Self and modifies the following attributes:
        TODO.
        """
        self._problem_kind = "linear"
        self._distributions = DistributionCollection()
        self._solution = None
        self._policy_key = policy_key
        try:
            self._distribution_id = pd.Series(self.adata.obs[policy_key])
        except KeyError:
            raise KeyError(f"Unable to find data in `adata.obs[{policy_key!r}]`.") from None

        self._policy = create_policy(policy, adata=self.adata, key=policy_key)
        if isinstance(self._policy, ExplicitPolicy):
            self._policy = self._policy.create_graph(subset=subset)
        elif isinstance(self._policy, StarPolicy):
            self._policy = self._policy.create_graph(reference=reference)
        else:
            _ = self.policy.create_graph()  # type: ignore[union-attr]
        self._sample_pairs = list(self.policy._graph)  # type: ignore[union-attr]

        for el in self.policy.categories:  # type: ignore[union-attr]
            adata_masked = self.adata[self._create_mask(el)]
            a_created = self._create_marginals(adata_masked, data=a, source=True, **kwargs)
            b_created = self._create_marginals(adata_masked, data=b, source=False, **kwargs)
            self.distributions[el] = DistributionContainer.from_adata(  # type: ignore[index]
                adata_masked, a=a_created, b=b_created, **xy, **xx, **conditions
            )
        return self

    @wrap_solve
    def solve(
        self,
        backend: Literal["ott"] = "ott",
        solver_name: Literal["GENOTLinSolver"] = "GENOTLinSolver",
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> "CondOTProblem":
        """Solve optimal transport problem.

        Parameters
        ----------
        backend
            Which backend to use, see :func:`moscot.backends.utils.get_available_backends`.
        device
            Device where to transfer the solution, see :meth:`moscot.base.output.BaseDiscreteSolverOutput.to`.
        kwargs
            Keyword arguments for :meth:`moscot.base.solver.BaseSolver.__call__`.


        Returns
        -------
        Self and modifies the following attributes:
        - :attr:`solver`: optimal transport solver.
        - :attr:`solution`: optimal transport solution.
        """
        tmp = next(iter(self.distributions))  # type: ignore[arg-type]
        input_dim = self.distributions[tmp].xy.shape[1]  # type: ignore[union-attr, index]
        cond_dim = self.distributions[tmp].conditions.shape[1]  # type: ignore[union-attr, index]

        solver_class = backends.get_solver(
            self.problem_kind, solver_name=solver_name, backend=backend, return_class=True
        )
        init_kwargs, call_kwargs = solver_class._partition_kwargs(**kwargs)
        self._solver = solver_class(input_dim=input_dim, cond_dim=cond_dim, **init_kwargs)
        # note that the solver call consists of solver._prepare and solver._solve
        sample_pairs = self._sample_pairs if self._sample_pairs is not None else []
        self._solution = self._solver(  # type: ignore[misc]
            device=device,
            distributions=self.distributions,
            sample_pairs=self._sample_pairs,
            is_conditional=len(sample_pairs) > 1,
            **call_kwargs,
        )

        return self

    def _create_marginals(
        self, adata: AnnData, *, source: bool, data: Optional[str] = None, **kwargs: Any
    ) -> ArrayLike:
        if data is True:
            marginals = self.estimate_marginals(adata, source=source, **kwargs)
        elif data in (False, None):
            marginals = np.ones((adata.n_obs,), dtype=float) / adata.n_obs
        elif isinstance(data, str):
            try:
                marginals = np.asarray(adata.obs[data], dtype=float)
            except KeyError:
                raise KeyError(f"Unable to find data in `adata.obs[{data!r}]`.") from None
        return marginals

    def _create_mask(self, value: Union[K, Sequence[K]], *, allow_empty: bool = False) -> ArrayLike:
        """Create a mask used to subset the data.

        TODO(@MUCDK): this is copied from SubsetPolicy, consider making this a function.

        Parameters
        ----------
        value
            Values in the data which determine the mask.
        allow_empty
            Whether to allow empty mask.

        Returns
        -------
        Boolean mask of the same shape as the data.
        """
        if isinstance(value, str) or not isinstance(value, Iterable):
            mask = self._distribution_id == value
        else:
            mask = self._distribution_id.isin(value)
        if not allow_empty and not np.sum(mask):
            raise ValueError("Unable to construct an empty mask, use `allow_empty=True` to override.")
        return np.asarray(mask)

    @property
    def distributions(self) -> Optional[DistributionCollection[K]]:
        """Collection of distributions."""
        return self._distributions

    @property
    def adata(self) -> AnnData:
        """Source annotated data object."""
        return self._adata

    @property
    def solution(self) -> Optional[BaseDiscreteSolverOutput]:
        """Solution of the optimal transport problem."""
        return self._solution

    @property
    def solver(self) -> Optional[OTSolver[BaseDiscreteSolverOutput]]:
        """Solver of the optimal transport problem."""
        return self._solver

    @property
    def policy(self) -> Optional[SubsetPolicy[Any]]:
        """Policy used to subset the data."""
        return self._policy

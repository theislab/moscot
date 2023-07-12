import abc
import pathlib
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import cloudpickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandas.api import types as pd_types
from sklearn.preprocessing import StandardScaler

import scanpy as sc
from anndata import AnnData

from moscot import backends
from moscot._logging import logger
from moscot._types import ArrayLike, CostFn_t, Device_t, ProblemKind_t
from moscot.base.output import BaseSolverOutput, MatrixSolverOutput
from moscot.base.problems._utils import require_solution, wrap_prepare, wrap_solve
from moscot.base.solver import OTSolver
from moscot.utils.tagged_array import Tag, TaggedArray

__all__ = ["BaseProblem", "OTProblem"]


class BaseProblem(abc.ABC):
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

        self._solver: Optional[OTSolver[BaseSolverOutput]] = None
        self._solution: Optional[BaseSolverOutput] = None

        self._x: Optional[TaggedArray] = None
        self._y: Optional[TaggedArray] = None
        self._xy: Optional[TaggedArray] = None

        self._a: Optional[ArrayLike] = None
        self._b: Optional[ArrayLike] = None

    def _handle_linear(self, cost: CostFn_t = None, **kwargs: Any) -> TaggedArray:
        if "x_attr" not in kwargs or "y_attr" not in kwargs:
            kwargs.setdefault("tag", Tag.COST_MATRIX)
            attr = kwargs.pop("attr", "obsm")

            if attr in ("obsm", "uns"):
                return TaggedArray.from_adata(
                    self.adata_src, dist_key=(self._src_key, self._tgt_key), attr=attr, cost="custom", **kwargs
                )
            raise ValueError(f"Storing `{kwargs['tag']!r}` in `adata.{attr}` is disallowed.")

        x_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("x_")}
        y_kwargs = {k[2:]: v for k, v in kwargs.items() if k.startswith("y_")}
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
        xy: Union[Mapping[str, Any], TaggedArray] = types.MappingProxyType({}),
        x: Union[Mapping[str, Any], TaggedArray] = types.MappingProxyType({}),
        y: Union[Mapping[str, Any], TaggedArray] = types.MappingProxyType({}),
        a: Optional[Union[bool, str, ArrayLike]] = None,
        b: Optional[Union[bool, str, ArrayLike]] = None,
        **kwargs: Any,
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
        kwargs
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
        if xy and not x and not y:
            self._problem_kind = "linear"
            self._xy = xy if isinstance(xy, TaggedArray) else self._handle_linear(**xy)
        elif x and y and not xy:
            self._problem_kind = "quadratic"
            if isinstance(x, TaggedArray):
                self._x = x
            else:
                self._x = TaggedArray.from_adata(self.adata_src, dist_key=self._src_key, **x)
            if isinstance(y, TaggedArray):
                self._y = y
            else:
                self._y = TaggedArray.from_adata(self.adata_tgt, dist_key=self._tgt_key, **y)
        elif xy and x and y:
            self._problem_kind = "quadratic"
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

    @wrap_solve
    def solve(
        self,
        backend: Literal["ott"] = "ott",
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> "OTProblem":
        """Solve the :term:`OT` problem.

        Parameters
        ----------
        backend
            Which backend to use, see :func:`~moscot.backends.utils.get_available_backends`.
        device
            Transfer the solution to a different device, see :meth:`~moscot.base.output.BaseSolverOutput.to`.
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
        solver_class = backends.get_solver(self.problem_kind, backend=backend, return_class=True)
        init_kwargs, call_kwargs = solver_class._partition_kwargs(**kwargs)
        self._solver = solver_class(**init_kwargs)

        self._solution = self._solver(  # type: ignore[misc]
            xy=self._xy,
            x=self._x,
            y=self._y,
            a=self.a,
            b=self.b,
            device=device,
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
        r"""Push data through the :attr:`~moscot.base.output.BaseSolverOutput.transport_matrix`.

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
            assert isinstance(self.solution, BaseSolverOutput)
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
        r"""Pull data through the :attr:`~moscot.base.output.BaseSolverOutput.transport_matrix`.

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
            assert isinstance(self.solution, BaseSolverOutput)
        data = self._get_mass(self.adata_tgt, data=data, subset=subset, normalize=normalize, split_mass=split_mass)
        return self.solution.pull(data, scale_by_marginals=scale_by_marginals)

    def set_solution(
        self, solution: Union[ArrayLike, pd.DataFrame, BaseSolverOutput], *, overwrite: bool = False, **kwargs: Any
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
            pd.testing.assert_series_equal(self.adata_src.obs_names.to_series(), solution.index.to_series())
            pd.testing.assert_series_equal(self.adata_tgt.obs_names.to_series(), solution.columns.to_series())
            solution = solution.to_numpy()
        if not isinstance(solution, BaseSolverOutput):
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
    ) -> Dict[Literal["xy", "x", "y"], TaggedArray]:
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
                return {"xy": TaggedArray(data[:n], data[n:], tag=Tag.POINT_CLOUD)}

            logger.info(f"Computing pca with `n_comps={n_comps}` for `xy` using `{msg}`")
            data = sc.pp.pca(data, n_comps=n_comps, **kwargs)
            if scaler is not None:
                data = scaler.fit_transform(data)
            return {"xy": TaggedArray(data[:n], data[n:], tag=Tag.POINT_CLOUD)}
        if term in ("x", "y"):  # if we don't have a shared space, then adata_y is always None
            logger.info(f"Computing pca with `n_comps={n_comps}` for `{term}` using `{msg}`")
            x = sc.pp.pca(x, n_comps=n_comps, **kwargs)
            if scaler is not None:
                x = scaler.fit_transform(x)
            return {term: TaggedArray(x, tag=Tag.POINT_CLOUD)}
        raise ValueError(f"Expected `term` to be one of `x`, `y`, or `xy`, found `{term!r}`.")

    @staticmethod
    def _spatial_norm_callback(
        term: Literal["x", "y"],
        adata: AnnData,
        adata_y: Optional[AnnData] = None,
        **kwargs: Any,
    ) -> Dict[Literal["x", "y"], TaggedArray]:
        spatial_key = kwargs["spatial_key"]
        if term == "x":
            spatial = adata.obsm[spatial_key]
        if term == "y":
            if adata_y is None:
                raise ValueError("When `term` is `y`, `adata_y` cannot be `None`.")
            spatial = adata_y.obsm[spatial_key]

        logger.info(f"Normalizing spatial coordinates of `{term}`.")
        spatial = (spatial - spatial.mean()) / spatial.std()
        return {term: TaggedArray(spatial, tag=Tag.POINT_CLOUD)}

    def _create_marginals(
        self,
        adata: AnnData,
        *,
        source: bool,
        data: Optional[Union[bool, str, ArrayLike]] = None,
        marginal_kwargs: Dict[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> ArrayLike:
        if data is True:
            marginals = self.estimate_marginals(adata, source=source, **marginal_kwargs, **kwargs)
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

    def estimate_marginals(self, adata: AnnData, *, source: bool, **kwargs: Any) -> ArrayLike:
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
        del kwargs
        return np.ones((adata.n_obs,), dtype=float) / adata.n_obs

    # TODO(michalk8): extend for point-clouds as Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
    # TODO(michalk8): allow this to be nullified
    def set_xy(
        self,
        data: pd.DataFrame,
        tag: Literal["cost", "kernel"],
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
        pd.testing.assert_series_equal(self.adata_src.obs_names.to_series(), data.index.to_series())
        pd.testing.assert_series_equal(self.adata_tgt.obs_names.to_series(), data.columns.to_series())

        self._xy = TaggedArray(data_src=data.to_numpy(), data_tgt=None, tag=Tag(tag), cost="cost")
        self._stage = "prepared"

    def set_x(self, data: pd.DataFrame, tag: Literal["cost", "kernel"]) -> None:
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
        pd.testing.assert_series_equal(self.adata_src.obs_names.to_series(), data.index.to_series())
        pd.testing.assert_series_equal(self.adata_src.obs_names.to_series(), data.columns.to_series())

        if self.problem_kind == "linear":
            logger.info(f"Changing the problem type from {self.problem_kind!r} to 'quadratic (fused)'.")
            self._problem_kind = "quadratic"
        self._x = TaggedArray(data_src=data.to_numpy(), data_tgt=None, tag=Tag(tag), cost="cost")
        self._stage = "prepared"

    def set_y(self, data: pd.DataFrame, tag: Literal["cost", "kernel"]) -> None:
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
        pd.testing.assert_series_equal(self.adata_tgt.obs_names.to_series(), data.index.to_series())
        pd.testing.assert_series_equal(self.adata_tgt.obs_names.to_series(), data.columns.to_series())

        if self.problem_kind == "linear":
            logger.info(f"Changing the problem type from {self.problem_kind!r} to 'quadratic (fused)'.")
            self._problem_kind = "quadratic"
        self._y = TaggedArray(data_src=data.to_numpy(), data_tgt=None, tag=Tag(tag), cost="cost")
        self._stage = "prepared"

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
    def solution(self) -> Optional[BaseSolverOutput]:
        """Solution of the :term:`OT` problem."""
        return self._solution

    @property
    def solver(self) -> Optional[OTSolver[BaseSolverOutput]]:
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

import abc
import types
from typing import (
    Any,
    Dict,
    Generic,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from moscot._logging import logger
from moscot._types import ArrayLike, Device_t, ProblemKind_t
from moscot.base.output import BaseSolverOutput
from moscot.utils.tagged_array import Tag, TaggedArray

__all__ = ["BaseSolver", "OTSolver"]


O = TypeVar("O", bound=BaseSolverOutput)


class TaggedArrayData(NamedTuple):  # noqa: D101
    x: Optional[TaggedArray]
    y: Optional[TaggedArray]
    xy: Optional[TaggedArray]


class TagConverter:  # noqa: D101
    def _get_array_data(
        self,
        xy: Optional[Union[TaggedArray, ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        x: Optional[Union[TaggedArray, ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        y: Optional[Union[TaggedArray, ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        tags: Mapping[Literal["xy", "x", "y"], Tag] = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> TaggedArrayData:
        def to_tuple(
            data: Optional[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]]
        ) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
            if not isinstance(data, tuple):
                return data, None
            if len(data) != 2:
                raise ValueError(f"Expected data to be of length `2`, found `{len(data)}`.")
            return data

        loss_xy = {k[3:]: v for k, v in kwargs.items() if k.startswith("xy_")}
        loss_x = {k[2:]: v for k, v in kwargs.items() if k.startswith("x_")}
        loss_y = {k[2:]: v for k, v in kwargs.items() if k.startswith("y_")}

        # fmt: off
        xy = xy if isinstance(xy, TaggedArray) else self._convert(*to_tuple(xy), tag=tags.get("xy", None), **loss_xy)
        x = x if isinstance(x, TaggedArray) else self._convert(*to_tuple(x), tag=tags.get("x", None), **loss_x)
        y = y if isinstance(y, TaggedArray) else self._convert(*to_tuple(y), tag=tags.get("y", None), **loss_y)
        # fmt: on

        return TaggedArrayData(x=x, y=y, xy=xy)

    @staticmethod
    def _convert(
        x: Optional[ArrayLike] = None, y: Optional[ArrayLike] = None, *, tag: Optional[Tag] = None, **kwargs: Any
    ) -> Optional[TaggedArray]:
        if x is None:
            return None  # data not needed; checks are done later

        if y is None:
            if tag is None:
                tag = Tag.POINT_CLOUD
                logger.info(f"Unspecified tag for `x`. Using `tag={tag!r}`")
            if tag == Tag.POINT_CLOUD:
                y = x
        else:  # always a point cloud
            if tag is None:
                tag = Tag.POINT_CLOUD
            if tag != Tag.POINT_CLOUD:
                logger.warning(f"Unable to handle `tag={tag!r}` for `y`. Using `tag={Tag.POINT_CLOUD!r}`")
                tag = Tag.POINT_CLOUD

        return TaggedArray(data_src=x, data_tgt=y, tag=tag, **kwargs)


class BaseSolver(Generic[O], abc.ABC):
    """Base class for all solvers."""

    @abc.abstractmethod
    def _prepare(self, **kwargs: Any) -> Any:
        """Prepare a problem.

        Parameters
        ----------
        kwargs
            Keyword arguments.

        Returns
        -------
        Object passed to :meth:`_solve`.
        """

    @abc.abstractmethod
    def _solve(self, data: Any, **kwargs: Any) -> O:
        """Solve a problem.

        Parameters
        ----------
        data
            Object returned by :meth:`_prepare`.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        The output.
        """

    @property
    @abc.abstractmethod
    def problem_kind(self) -> ProblemKind_t:
        """Problem kind this solver handles."""

    def __call__(self, **kwargs: Any) -> O:
        """Solve a problem.

        Parameters
        ----------
        kwargs
            Keyword arguments for data preparation.

        Returns
        -------
        The solver output.
        """
        data = self._prepare(**kwargs)
        return self._solve(data)

    @classmethod
    @abc.abstractmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        """Return arguments specific for :meth:`__call__` and arguments shared with :class:`BaseSolver`."""

    @classmethod
    def _partition_kwargs(cls, **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Partition keyword arguments.

        Used by the :meth:`~moscot.problems.base.BaseProblem.solve`.

        Parameters
        ----------
        kwargs
            Keyword arguments to partition.

        Returns
        -------
        Keyword arguments for :class:`BaseSolver` and :meth:`__call__`, respectively.
        """
        call_kws, shared_kws = cls._call_kwargs()
        init_kwargs = {k: v for k, v in kwargs.items() if k not in call_kws or k in shared_kws}
        call_kwargs = {k: v for k, v in kwargs.items() if k in call_kws or k in shared_kws}

        return init_kwargs, call_kwargs


class OTSolver(TagConverter, BaseSolver[O], abc.ABC):
    """Base class for optimal transport solvers."""

    def __call__(
        self,
        xy: Optional[Union[TaggedArray, ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        x: Optional[Union[TaggedArray, ArrayLike]] = None,
        y: Optional[Union[TaggedArray, ArrayLike]] = None,
        tags: Mapping[Literal["x", "y", "xy"], Tag] = types.MappingProxyType({}),
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> O:
        """Solve an optimal transport problem.

        Parameters
        ----------
        xy
            Data that defines the linear term.
        x
            Data of the first geometry that defines the quadratic term.
        y
            Data of the second geometry that defines the quadratic term.
        tags
            How to interpret the data in ``xy``, ``x`` and ``y``.
        device
            Device to transfer the output to, see :meth:`~moscot.base.output.BaseSolverOutput.to`.
        kwargs
            Keyword arguments for parent's :meth:`__call__`.

        Returns
        -------
        The optimal transport solution.
        """
        data = self._get_array_data(xy=xy, x=x, y=y, tags=tags)
        kwargs = {**kwargs, **self._untag(data)}
        res = super().__call__(**kwargs)
        if not res.converged:
            logger.warning("Solver did not converge")

        return res.to(device=device)  # type: ignore[return-value]

    def _untag(self, data: TaggedArrayData) -> Dict[str, Any]:
        if self.problem_kind == "linear":
            if data.xy is None:
                raise ValueError("No data specified for the linear term.")
            data_kwargs: Dict[str, Any] = {"xy": data.xy}
        elif self.problem_kind == "quadratic":
            if data.x is None or data.y is None:
                raise ValueError("No data specified for the quadratic term.")
            # `data.xy` can be `None`, in which case GW is used
            data_kwargs = {"x": data.x, "y": data.y, "xy": data.xy}
        else:
            raise NotImplementedError(f"Unable to prepare data for `{self.problem_kind}` problem.")

        return data_kwargs

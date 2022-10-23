from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Dict, Tuple, Union, Generic, Literal, Mapping, TypeVar, Optional, NamedTuple
import warnings

from moscot._types import Device_t, ArrayLike
from moscot._logging import logger
from moscot._docs._docs import d
from moscot.solvers._output import BaseSolverOutput
from moscot._constants._enum import ModeEnum
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ["ProblemKind", "BaseSolver", "OTSolver"]


O = TypeVar("O", bound=BaseSolverOutput)


class ProblemKind(ModeEnum):
    """Class defining the problem class and dispatching the solvers."""

    UNKNOWN = "unknown"
    LINEAR = "linear"
    QUAD = "quadratic"
    QUAD_FUSED = "quadratic_fused"

    def solver(self, *, backend: Literal["ott"] = "ott", **kwargs: Any) -> "BaseSolver[O]":
        """
        Return the solver dependent on the backend and the problem type.

        Parameters
        ----------
        backend
            The backend the solver corresponds to.

        Returns
        -------
        Solver corresponding to the backend and problem type.
        """
        if backend == "ott":
            from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver  # type: ignore[attr-defined]

            if self == ProblemKind.LINEAR:
                return SinkhornSolver(**kwargs)
            if self == ProblemKind.QUAD:
                return GWSolver(**kwargs)
            if self == ProblemKind.QUAD_FUSED:
                return FGWSolver(**kwargs)
            raise NotImplementedError(f"TODO: {self}")

        raise NotImplementedError(f"Invalid backend: `{backend}`")


class TaggedArrayData(NamedTuple):
    x: Optional[TaggedArray]
    y: Optional[TaggedArray]
    xy: Optional[TaggedArray]


class TagConverterMixin:
    def _get_array_data(
        self,
        xy: Optional[Union[TaggedArray, ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        x: Optional[Union[TaggedArray, ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        y: Optional[Union[TaggedArray, ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        tags: Mapping[Literal["xy", "x", "y"], Tag] = MappingProxyType({}),
        **kwargs: Any,
    ) -> TaggedArrayData:
        def to_tuple(
            data: Optional[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]]
        ) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
            if not isinstance(data, tuple):
                return data, None
            if len(data) != 2:
                raise ValueError(f"TODO: `{data}`")
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
                logger.info(f"TODO: unspecified tag`, using `{tag}`")
            if tag == Tag.POINT_CLOUD:
                y = x
        else:  # always a point cloud
            if tag is None:
                tag = Tag.POINT_CLOUD
            if tag != Tag.POINT_CLOUD:
                logger.info(f"TODO: specified `{tag}`, using `{Tag.POINT_CLOUD}`")
                tag = Tag.POINT_CLOUD

        return TaggedArray(x, y, tag=tag, **kwargs)


class BaseSolver(Generic[O], ABC):
    """BaseSolver class."""

    @abstractmethod
    def _prepare(self, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _solve(self, data: Any, **kwargs: Any) -> O:
        pass

    @property
    @abstractmethod
    def problem_kind(self) -> ProblemKind:
        """Problem kind."""
        # helps to check whether necessary inputs were passed

    def __call__(self, **kwargs: Any) -> O:
        """Call method."""
        data = self._prepare(**kwargs)
        return self._solve(data)


@d.get_sections(base="OTSolver", sections=["Parameters", "Raises"])
@d.dedent
class OTSolver(TagConverterMixin, BaseSolver[O], ABC):  # noqa: B024
    """OTSolver class."""

    def __call__(
        self,
        xy: Optional[Union[TaggedArray, ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        x: Optional[Union[TaggedArray, ArrayLike]] = None,
        y: Optional[Union[TaggedArray, ArrayLike]] = None,
        tags: Mapping[Literal["x", "y", "xy"], Tag] = MappingProxyType({}),
        device: Optional[Device_t] = None,
        **kwargs: Any,
    ) -> O:
        """Call method."""
        data = self._get_array_data(xy=xy, x=x, y=y, tags=tags)
        kwargs = self._prepare_kwargs(data, **kwargs)
        res = super().__call__(**kwargs)
        if not res.converged:  # TODO use logging, not warnings
            warnings.warn("Solver did not converge")

        return res.to(device=device)  # type: ignore[return-value]

    def _prepare_kwargs(self, data: TaggedArrayData, **kwargs: Any) -> Dict[str, Any]:
        def assert_linear() -> None:
            if data.xy is None:
                raise ValueError("TODO: no linear data.")

        def assert_quadratic() -> None:
            if data.x is None or data.y is None:
                raise ValueError("TODO: no quadratic data.")

        if self.problem_kind == ProblemKind.LINEAR:
            assert_linear()
            data_kwargs: Dict[str, Any] = {"xy": data.xy}
        elif self.problem_kind == ProblemKind.QUAD:
            assert_quadratic()
            data_kwargs = {"x": data.x, "y": data.y}
        elif self.problem_kind == ProblemKind.QUAD_FUSED:
            assert_linear()
            assert_quadratic()
            data_kwargs = {"x": data.x, "y": data.y, "xy": data.xy}
        else:
            raise NotImplementedError(f"TODO: {self.problem_kind}")

        return {**kwargs, **data_kwargs}

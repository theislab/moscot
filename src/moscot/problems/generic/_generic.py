import types
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._docs._docs import d
from moscot._types import (
    CostKwargs_t,
    OttCostFn_t,
    OttCostFnMap_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
    SinkhornInitializer_t,
)
from moscot.base.problems.compound_problem import B, CompoundProblem, K
from moscot.base.problems.problem import OTProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.generic._mixins import GenericAnalysisMixin

__all__ = ["SinkhornProblem", "GWProblem"]


@d.dedent
class SinkhornProblem(GenericAnalysisMixin[K, B], CompoundProblem[K, B]):
    """
    Class for solving linear OT problems.

    Parameters
    ----------
    %(adata)s
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        cost: OttCostFn_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "SinkhornProblem[K, B]":
        """
        Prepare the :class:`moscot.problems.generic.SinkhornProblem`.

        Parameters
        ----------
        %(key)s
        %(joint_attr)s
        %(policy)s
        %(cost_lin)s
        %(cost_kwargs)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.generic.SinkhornProblem`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.

        Examples
        --------
        %(ex_prepare)s
        """
        self.batch_key = key  # type: ignore[misc]
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, _, _ = handle_cost(xy=xy, cost=cost, cost_kwargs=cost_kwargs)
        return super().prepare(
            key=key,
            policy=policy,
            xy=xy,
            cost=cost,
            a=a,
            b=b,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        epsilon: Optional[float] = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: SinkhornInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        threshold: float = 1e-3,
        lse_mode: bool = True,
        norm_error: int = 1,
        inner_iterations: int = 10,
        min_iterations: int = 0,
        max_iterations: int = 2000,
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        cost_matrix_rank: Optional[int] = None,
        **kwargs: Any,
    ) -> "SinkhornProblem[K,B]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.generic.SinkhornProblem`.

        Parameters
        ----------
        %(epsilon)s
        %(tau_a)s
        %(tau_b)s
        %(rank)s
        %(scale_cost)s
        %(pointcloud_kwargs)s
        %(stage)s
        %(initializer_lin)s
        %(initializer_kwargs)s
        %(jit)s
        %(sinkhorn_kwargs)s
        %(device_solve)s
        %(cost_matrix_rank)s
        %(kwargs_linear)s

        Returns
        -------
        :class:`moscot.problems.generic.SinkhornProblem`.

        Examples
        --------
        %(ex_solve_linear)s
        """
        return super().solve(  # type: ignore[return-value]
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            rank=rank,
            scale_cost=scale_cost,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            jit=jit,
            threshold=threshold,
            lse_mode=lse_mode,
            norm_error=norm_error,
            inner_iterations=inner_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            cost_matrix_rank=cost_matrix_rank,
            device=device,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.PAIRWISE, _constants.EXPLICIT  # type: ignore[return-value]


@d.get_sections(base="GWProblem", sections=["Parameters"])
@d.dedent
class GWProblem(GenericAnalysisMixin[K, B], CompoundProblem[K, B]):
    """
    Class for solving (Fused) Gromov-Wasserstein problems.

    Parameters
    ----------
    %(adata)s
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        key: str,
        x_attr: Union[str, Mapping[str, Any]],
        y_attr: Union[str, Mapping[str, Any]],
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        cost: OttCostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "GWProblem[K, B]":
        """
        Prepare the :class:`moscot.problems.generic.GWProblem`.

        Parameters
        ----------
        %(key)s
        %(x_attr)s
        %(y_attr)s
        %(joint_attr)s
        %(policy)s
        %(cost)s
        %(cost_kwargs)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.generic.GWProblem`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.

        Examples
        --------
        %(ex_prepare)s
        """
        self.batch_key = key  # type: ignore[misc]

        def set_quad_defaults(z: Union[str, Mapping[str, Any]]) -> Dict[str, str]:
            if isinstance(z, str):
                return {"attr": "obsm", "key": z, "tag": "point_cloud"}  # cost handled by handle_cost
            if isinstance(z, Mapping):
                return dict(z)
            raise TypeError("`x_attr` and `y_attr` must be of type `str` or `dict`.")

        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        x = set_quad_defaults(x_attr)
        y = set_quad_defaults(y_attr)
        xy, x, y = handle_cost(xy=xy, x=x, y=y, cost=cost, cost_kwargs=cost_kwargs)
        return super().prepare(
            key=key,
            xy=xy,
            x=x,
            y=y,
            policy=policy,
            cost=cost,
            a=a,
            b=b,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        alpha: float = 1.0,
        epsilon: Optional[float] = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "GWProblem[K,B]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.generic.GWProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(tau_a)s
        %(tau_b)s
        %(rank)s
        %(scale_cost)s
        %(pointcloud_kwargs)s
        %(stage)s
        %(initializer_quad)s
        %(initializer_kwargs)s
        %(gw_kwargs)s
        %(linear_solver_kwargs)s
        %(device_solve)s
        %(kwargs_quad)s

        Returns
        -------
        :class:`moscot.problems.generic.GWProblem`.

        Examples
        --------
        %(ex_solve_quadratic)s
        """
        return super().solve(  # type: ignore[return-value]
            alpha=alpha,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            rank=rank,
            scale_cost=scale_cost,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            jit=jit,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            linear_solver_kwargs=linear_solver_kwargs,
            device=device,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.PAIRWISE, _constants.EXPLICIT  # type: ignore[return-value]

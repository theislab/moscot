import types
from typing import Any, Literal, Mapping, Optional, Tuple, Type, Union

from moscot import _constants
from moscot._docs._docs import d
from moscot._types import (
    CostKwargs_t,
    OttCostFnMap_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
)
from moscot.base.problems.compound_problem import B, CompoundProblem, K
from moscot.base.problems.problem import OTProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.space._mixins import SpatialAlignmentMixin

__all__ = ["AlignmentProblem"]


@d.dedent
class AlignmentProblem(CompoundProblem[K, B], SpatialAlignmentMixin[K, B]):
    """
    Class for aligning spatial omics data, based on :cite:`zeira:22`.

    The `AlignmentProblem` allows to align spatial omics data via optimal transport.

    Parameters
    ----------
    %(adata)s
    """

    @d.dedent
    def prepare(
        self,
        batch_key: str,
        spatial_key: str = "spatial",
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "star"] = "sequential",
        reference: Optional[str] = None,
        cost: OttCostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "AlignmentProblem[K, B]":
        """Prepare the problem.

        This method prepares the data to be passed to the optimal transport solver.

        Parameters
        ----------
        %(batch_key)s
        %(spatial_key)s
        %(joint_attr)s
        %(policy)s

        reference
            Only used if `policy="star"`, it's the value for reference stored
            in :attr:`anndata.AnnData.obs` ``["batch_key"]``.

        %(cost)s
        %(cost_kwargs)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`.

        Examples
        --------
        %(ex_prepare)s
        """
        self.spatial_key = spatial_key
        self.batch_key = batch_key

        x = y = {"attr": "obsm", "key": self.spatial_key, "tag": "point_cloud"}

        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, x, y = handle_cost(xy=xy, x=x, y=y, cost=cost, cost_kwargs=cost_kwargs)

        return super().prepare(
            x=x, y=y, xy=xy, policy=policy, key=batch_key, reference=reference, cost=cost, a=a, b=b, **kwargs
        )

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-2,
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
    ) -> "AlignmentProblem[K,B]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.AlignmentProblem`.

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
        %(kwargs_quad_fused)s

        Returns
        -------
        :class:`moscot.problems.space.AlignmentProblem`.

        Examples
        --------
        %(ex_solve_quadratic)s
        """
        return super().solve(
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
        )  # type: ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.STAR  # type: ignore[return-value]

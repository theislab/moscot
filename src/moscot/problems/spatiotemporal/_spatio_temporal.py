import types
from typing import Any, Literal, Mapping, Optional, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._docs._docs import d
from moscot._types import (
    CostKwargs_t,
    Numeric_t,
    OttCostFnMap_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
)
from moscot.base.problems.birth_death import BirthDeathMixin, BirthDeathProblem
from moscot.base.problems.compound_problem import B
from moscot.problems.space import AlignmentProblem, SpatialAlignmentMixin
from moscot.problems.time import TemporalMixin

__all__ = ["SpatioTemporalProblem"]


@d.dedent
class SpatioTemporalProblem(
    TemporalMixin[Numeric_t, BirthDeathProblem],
    BirthDeathMixin,
    AlignmentProblem[Numeric_t, BirthDeathProblem],
    SpatialAlignmentMixin[Numeric_t, BirthDeathProblem],
):
    """
    Class for analyzing time series spatial single cell data.

    The `SpatioTemporalProblem` allows to model and analyze spatio-temporal single cell data
    by matching cells belonging to two different time points via OT.

    Parameters
    ----------
    %(adata)s
    """

    # TODO(michalk8): check if this is necessary
    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        time_key: str,
        spatial_key: str = "spatial",
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        cost: OttCostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        marginal_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> "SpatioTemporalProblem":
        """Prepare the problem.

        This method executes multiple steps to prepare the problem for the Optimal Transport solver to be ready
        to solve it.

        Parameters
        ----------
        %(time_key)s
        %(spatial_key)s
        %(joint_attr)s
        %(policy)s
        %(cost)s
        %(cost_kwargs)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        The prepared problem.

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.

        Examples
        --------
        %(ex_prepare)s
        """
        # spatial key set in AlignmentProblem
        # handle_joint_attr and handle_cost in AlignmentProblem
        self.temporal_key = time_key
        # TODO(michalk8): needs to be modified, move into BirthDeathMixin?
        marginal_kwargs = dict(marginal_kwargs)
        marginal_kwargs["proliferation_key"] = self.proliferation_key
        marginal_kwargs["apoptosis_key"] = self.apoptosis_key
        if a is None:
            a = self.proliferation_key is not None or self.apoptosis_key is not None
        if b is None:
            b = self.proliferation_key is not None or self.apoptosis_key is not None

        return super().prepare(
            spatial_key=spatial_key,
            batch_key=time_key,
            joint_attr=joint_attr,
            policy=policy,
            reference=None,
            cost=cost,
            cost_kwargs=cost_kwargs,
            a=a,
            b=b,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
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
    ) -> "SpatioTemporalProblem":
        """Solve the problem.

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
        The solved problem.

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
        )

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return (
            _constants.SEQUENTIAL,
            _constants.TRIL,
            _constants.TRIU,
            _constants.EXPLICIT,
        )  # type: ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[B]:  # type: ignore[override]
        return BirthDeathProblem  # type: ignore[return-value]

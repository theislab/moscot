import types
from typing import Any, Literal, Mapping, Optional, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._docs._docs import d
from moscot._types import (
    CostFnMap_t,
    CostKwargs_t,
    Numeric_t,
    OttCostFn_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
    SinkhornInitializer_t,
)
from moscot.base.problems.birth_death import BirthDeathMixin, BirthDeathProblem
from moscot.base.problems.compound_problem import B, CompoundProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.time._mixins import TemporalMixin

__all__ = ["TemporalProblem", "LineageProblem"]


@d.dedent
class TemporalProblem(
    TemporalMixin[Numeric_t, BirthDeathProblem], BirthDeathMixin, CompoundProblem[Numeric_t, BirthDeathProblem]
):
    """
    Class for analyzing time series single cell data based on :cite:`schiebinger:19`.

    The `TemporalProblem` allows to model and analyze time series single cell data by matching
    cells from previous time points to later time points via OT.
    Based on the assumption that the considered cell modality is similar in consecutive time points
    probabilistic couplings are computed between different time points.
    This allows to understand cell trajectories by inferring ancestors and descendants of single cells.

    Parameters
    ----------
    %(adata)s

    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        time_key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        cost: OttCostFn_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        marginal_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Prepare the :class:`moscot.problems.time.TemporalProblem`.

        This method executes multiple steps to prepare the optimal transport problems.

        Parameters
        ----------
        %(time_key)s
        %(joint_attr)s
        %(policy)s
        %(cost_lin)s
        %(cost_kwargs)s
        %(a_temporal)s
        %(b_temporal)s
        %(marginal_kwargs)s
        %(kwargs_prepare)s


        Returns
        -------
        :class:`moscot.problems.time.TemporalProblem`.

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.

        Examples
        --------
        %(ex_prepare)s
        """
        self.temporal_key = time_key
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, x, y = handle_cost(xy=xy, x=kwargs.pop("x", {}), y=kwargs.pop("y", {}), cost=cost, cost_kwargs=cost_kwargs)

        # TODO(michalk8): needs to be modified
        marginal_kwargs = dict(marginal_kwargs)
        marginal_kwargs["proliferation_key"] = self.proliferation_key
        marginal_kwargs["apoptosis_key"] = self.apoptosis_key
        if a is None:
            a = self.proliferation_key is not None or self.apoptosis_key is not None
        if b is None:
            b = self.proliferation_key is not None or self.apoptosis_key is not None

        return super().prepare(
            key=time_key,
            xy=xy,
            x=x,
            y=y,
            policy=policy,
            cost=None,  # cost information is already stored in x,y,xy
            marginal_kwargs=marginal_kwargs,
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
        cost_matrix_rank: Optional[int] = None,
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.time.TemporalProblem`.

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
        :class:`moscot.problems.time.TemporalProblem`.

        Examples
        --------
        %(ex_solve_linear)s
        """
        return super().solve(
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
        )  # type:ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[B]:  # type: ignore[override]
        return BirthDeathProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.TRIU, _constants.EXPLICIT  # type: ignore[return-value]


@d.dedent
class LineageProblem(TemporalProblem):
    """
    Estimator for modelling time series single cell data based on moslin.

    Class handling the computation and downstream analysis of temporal single cell data with lineage prior.

    Parameters
    ----------
    %(adata)s
    """

    @d.dedent
    def prepare(
        self,
        time_key: str,
        lineage_attr: Mapping[str, Any] = types.MappingProxyType({}),
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "sequential"] = "sequential",
        # TODO(michalk8): update
        cost: CostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        marginal_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> "LineageProblem":
        """
        Prepare the :class:`moscot.problems.time.LineageProblem`.

        Parameters
        ----------
        %(time_key)s

        lineage_attr
            Specifies the way the lineage information is processed. TODO: Specify.

        %(joint_attr)s
        %(policy)s
        %(cost)s
        %(cost_kwargs)s
        %(a_temporal)s
        %(b_temporal)s
        %(marginal_kwargs)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.time.LineageProblem`

        Examples
        --------
        %(ex_prepare)s
        """
        if not len(lineage_attr) and ("cost_matrices" not in self.adata.obsp):
            raise KeyError("Unable to find cost matrices in `adata.obsp['cost_matrices']`.")

        x = y = lineage_attr

        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, x, y = handle_cost(xy=xy, x=x, y=y, cost=cost, cost_kwargs=cost_kwargs)

        x.setdefault("attr", "obsp")
        x.setdefault("key", "cost_matrices")
        x.setdefault("cost", "custom")
        x.setdefault("tag", "cost_matrix")

        y.setdefault("attr", "obsp")
        y.setdefault("key", "cost_matrices")
        y.setdefault("cost", "custom")
        y.setdefault("tag", "cost_matrix")

        return super().prepare(
            time_key,
            joint_attr=xy,
            x=x,
            y=y,
            policy=policy,
            cost=cost,
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
    ) -> "LineageProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.time.LineageProblem`.

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
        :class:`moscot.problems.time.LineageProblem`

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

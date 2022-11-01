from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional

from anndata import AnnData

from moscot._types import Numeric_t, ScaleCost_t, ProblemStage_t, QuadInitializer_t
from moscot._docs._docs import d
from moscot._constants._key import Key
from moscot._constants._constants import Policy
from moscot.problems.time._mixins import TemporalMixin
from moscot.problems.space._mixins import SpatialAlignmentMixin
from moscot.problems.space._alignment import AlignmentProblem
from moscot.problems.base._birth_death import BirthDeathMixin, BirthDeathProblem
from moscot.problems.base._compound_problem import B


@d.dedent
class SpatioTemporalProblem(
    TemporalMixin[Numeric_t, BirthDeathProblem],
    BirthDeathMixin,
    AlignmentProblem[Numeric_t, BirthDeathProblem],
    SpatialAlignmentMixin[Numeric_t, BirthDeathProblem],
):
    """Spatio-Temporal problem."""

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        time_key: str,
        spatial_key: str = Key.obsm.spatial,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        cost: Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "SpatioTemporalProblem":
        """
        Prepare the :class:`moscot.problems.spatio_temporal.SpatioTemporalProblem`.

        This method executes multiple steps to prepare the problem for the Optimal Transport solver to be ready
        to solve it

        Parameters
        ----------
        %(time_key)s
        %(spatial_key)s
        %(joint_attr)s
        %(policy)s
        %(cost)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.spatio_temporal.SpatioTemporalProblem`.

        Raises
        ------
        KeyError
            If `time_key` is not in :attr:`anndata.AnnData.obs`.
        KeyError
            If `spatial_key` is not in :attr:`anndata.AnnData.obs`.
        KeyError
            If `joint_attr` is a string and cannot be found in :attr:`anndata.AnnData.obsm`.
        ValueError
            If :attr:`adata.obsp` has no attribute `cost_matrices`.
        TypeError
            If `joint_attr` is not None, not a :class:`str` and not a :class:`dict`.

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.
        """
        # spatial key set in AlignmentProblem
        self.temporal_key = time_key

        return super().prepare(
            spatial_key=spatial_key,
            batch_key=time_key,
            joint_attr=joint_attr,
            policy=policy,
            reference=None,
            cost=cost,
            a=a,
            b=b,
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
        power: int = 1,
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        jit: bool = True,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        warm_start: Optional[bool] = None,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        gw_unbalanced_correction: bool = True,
        ranks: Union[int, Tuple[int, ...]] = -1,
        tolerances: Union[float, Tuple[float, ...]] = 1e-2,
        linear_solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "SpatioTemporalProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.SpatioTemporalProblem`.

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
        %(sinkhorn_lr_kwargs)s
        %(gw_lr_kwargs)s
        %(linear_solver_kwargs)s
        %(device_solve)s
        %(kwargs_quad_fused)s

        Returns
        -------
        :class:`moscot.problems.space.SpatioTemporalProblem`.
        """
        return super().solve(
            alpha=alpha,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            rank=rank,
            scale_cost=scale_cost,
            power=power,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            jit=jit,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            warm_start=warm_start,
            gamma=gamma,
            gamma_rescale=gamma_rescale,
            gw_unbalanced_correction=gw_unbalanced_correction,
            ranks=ranks,
            tolerances=tolerances,
            linear_solver_kwargs=linear_solver_kwargs,
            device=device,
            **kwargs,
        )

    @property
    def _valid_policies(self) -> Tuple[Policy, ...]:
        return Policy.SEQUENTIAL, Policy.TRIL, Policy.TRIU, Policy.EXPLICIT

    @property
    def _base_problem_type(self) -> Type[B]:
        return BirthDeathProblem  # type: ignore[return-value]

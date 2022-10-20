from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional

from anndata import AnnData

from moscot._types import Numeric_t, ScaleCost_t, QuadInitializer_t
from moscot._docs._docs import d
from moscot._constants._key import Key
from moscot._constants._constants import Policy, ScaleCost
from moscot.problems.time._mixins import TemporalMixin
from moscot.problems.space._mixins import SpatialAlignmentMixin
from moscot.problems.space._alignment import AlignmentProblem
from moscot.problems.base._birth_death import BirthDeathMixin, BirthDeathProblem
from moscot.problems.base._base_problem import ProblemStage
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
        joint_attr: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        policy: Literal[Policy.SEQUENTIAL, Policy.TRIL, Policy.TRIU, Policy.EXPLICIT] = Policy.SEQUENTIAL,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
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
        %(marginal_kwargs)s
        %(a)s
        %(b)s
        %(subset)s
        %(reference)s
        %(callback)s
        %(callback_kwargs)s

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
        policy = Policy(policy)  # type: ignore[assignment]
        marginal_kwargs = dict(marginal_kwargs)
        if self.proliferation_key is not None:
            marginal_kwargs["proliferation_key"] = self.proliferation_key
            kwargs["a"] = True
        if self.apoptosis_key is not None:
            marginal_kwargs["apoptosis_key"] = self.apoptosis_key
            kwargs["b"] = True

        return super().prepare(
            spatial_key=spatial_key,
            batch_key=time_key,
            joint_attr=joint_attr,
            policy=policy,
            reference=None,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-3,
        scale_cost: ScaleCost_t = ScaleCost.MEAN,
        stage: Union[ProblemStage, Tuple[ProblemStage, ...]] = (ProblemStage.PREPARED, ProblemStage.SOLVED),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "SpatioTemporalProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.SpatioTemporalProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(scale_cost)s
        %(rank)s
        %(stage)s
        %(initializer_quad)s
        %(initializer_kwargs)s
        %(solve_kwargs)s

        Returns
        -------
        :class:`moscot.problems.space.SpatioTemporalProblem`.
        """
        scale_cost = ScaleCost(scale_cost) if isinstance(scale_cost, ScaleCost) else scale_cost
        return super().solve(
            alpha=alpha,
            epsilon=epsilon,
            scale_cost=scale_cost,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            **kwargs,
        )

    @property
    def _valid_policies(self) -> Tuple[Policy, ...]:
        return Policy.SEQUENTIAL, Policy.TRIL, Policy.TRIU, Policy.EXPLICIT

    @property
    def _base_problem_type(self) -> Type[B]:
        return BirthDeathProblem  # type: ignore[return-value]

from typing import Any, Type, Tuple, Union, Mapping, Optional

from typing_extensions import Literal

from moscot._docs import d
from moscot._constants._key import Key
from moscot._constants._constants import Policy, ScaleCost
from moscot.problems.space._mixins import SpatialAlignmentMixin
from moscot.problems.base._base_problem import OTProblem, ScaleCost_t, ProblemStage
from moscot.problems.base._compound_problem import B, K, CompoundProblem

__all__ = ["AlignmentProblem"]


# need generic type B for SpatioTemporal
@d.dedent
class AlignmentProblem(CompoundProblem[K, B], SpatialAlignmentMixin[K, B]):
    """
    Class for aligning spatial omics data, based on :cite:`zeira2022`.

    The `AlignmentProblem` allows to align spatial omics data via optimal transport.

    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@giovp) LINK NOTEBOOK for how to use it
    """

    @d.dedent
    def prepare(
        self,
        batch_key: str,
        spatial_key: str = Key.obsm.spatial,
        joint_attr: Optional[Mapping[str, Any]] = None,
        policy: Literal[Policy.SEQUENTIAL, Policy.STAR] = Policy.SEQUENTIAL,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> "AlignmentProblem[K, B]":
        """
        Prepare the :class:`moscot.problems.space.AlignmentProblem`.

        This method prepares the data to be passed to the optimal transport solver.

        Parameters
        ----------
        %(batch_key)s
        %(spatial_key)s

        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps.
            - If None, the corresponding PCA space is computed.
            - If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of
            :class:`anndata.AnnData` as a key and the corresponding attribute as a value.

        %(policy)s

        reference
            Only used if `policy="star"`, it's the value for reference stored
            in :attr:`adata.obs` ``["batch_key"]``.

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`.
        """
        self.spatial_key = spatial_key
        self.batch_key = batch_key
        policy = Policy(policy)  # type: ignore[assignment]

        x = y = {"attr": "obsm", "key": self.spatial_key, "tag": "point_cloud"}

        if joint_attr is None:
            kwargs["callback"] = "local-pca"
            kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}
        return super().prepare(x=x, y=y, xy=joint_attr, policy=policy, key=batch_key, reference=reference, **kwargs)

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-3,
        scale_cost: ScaleCost_t = ScaleCost.MEAN,
        rank: int = -1,
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage, Tuple[ProblemStage, ...]] = (ProblemStage.PREPARED, ProblemStage.SOLVED),
        **kwargs: Any,
    ) -> "AlignmentProblem[K, B]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.AlignmentProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(scale_cost)s
        %(rank)s
        %(batch_size)s
        %(stage)s
        %(solve_kwargs)s

        Returns
        -------
        :class:`moscot.problems.space.AlignmentProblem`.
        """
        scale_cost = ScaleCost(scale_cost) if isinstance(scale_cost, ScaleCost) else scale_cost
        return super().solve(
            alpha=alpha, epsilon=epsilon, scale_cost=scale_cost, rank=rank, batch_size=batch_size, stage=stage, **kwargs
        )  # type:ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return Policy.SEQUENTIAL, Policy.STAR

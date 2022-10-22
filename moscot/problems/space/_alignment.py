from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional

from moscot._types import ScaleCost_t, ProblemStage_t, QuadInitializer_t
from moscot._docs._docs import d
from moscot._constants._key import Key
from moscot._constants._constants import Policy
from moscot.problems.space._mixins import SpatialAlignmentMixin
from moscot.problems.base._base_problem import OTProblem
from moscot.problems.base._compound_problem import B, K, CompoundProblem

__all__ = ["AlignmentProblem"]


@d.dedent
class AlignmentProblem(CompoundProblem[K, B], SpatialAlignmentMixin[K, B]):
    """
    Class for aligning spatial omics data, based on :cite:`zeira:22`.

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
        policy: Literal["sequential", "star"] = "sequential",
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
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        cost: Literal["Euclidean"] = "Euclidean",
        power: int = 1,
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        jit: bool = True,
        lse_mode: bool = True,
        norm_error: int = 1,
        inner_iterations: int = 10,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        warm_start: Optional[bool] = None,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        gw_unbalanced_correction: bool = True,
        ranks: Union[int, Tuple[int, ...]] = -1,
        tolerances: Union[float, Tuple[float, ...]] = 1e-2,
    ) -> "AlignmentProblem[K,B]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.time.AlignmentProblem`.

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

        Returns
        -------
        :class:`moscot.problems.space.AlignmentProblem`.
        """
        return super().solve(
            alpha=alpha,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            rank=rank,
            scale_cost=scale_cost,
            cost=cost,
            power=power,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            jit=jit,
            lse_mode=lse_mode,
            norm_error=norm_error,
            inner_iterations=inner_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            warm_start=warm_start,
            gamma=gamma,
            gamma_rescale=gamma_rescale,
            gw_unbalanced_correction=gw_unbalanced_correction,
            ranks=ranks,
            tolerances=tolerances,
        )  # type: ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return Policy.SEQUENTIAL, Policy.STAR

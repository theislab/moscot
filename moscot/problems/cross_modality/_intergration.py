from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional

from anndata import AnnData

from moscot._types import ScaleCost_t, ProblemStage_t, QuadInitializer_t
from moscot._docs._docs import d
from moscot._constants._constants import Policy
from moscot.problems.base._base_problem import OTProblem
from moscot.problems.base._compound_problem import B, K, CompoundProblem
from moscot.problems.cross_modality._intergration import IntegrationMixin  # type: ignore[attr-defined]

__all__ = ["IntegrationProblem"]


@d.dedent
class IntegrationProblem(CompoundProblem[K, OTProblem], IntegrationMixin[K, OTProblem]):
    """
    IntegrationProblem.

    Parameters
    ----------
    adata_modality_1
        Instance of :class:`anndata.AnnData` containing single cell data of modality 1.
    adata_modality_2
        Instance of :class:`anndata.AnnData` containing single cell data of modality 2.
    """

    def __init__(self, adata_1: AnnData, adata_2: AnnData, **kwargs: Any):
        super().__init__(adata_1, **kwargs)
        self._adata_2 = adata_2

    @d.dedent
    def prepare(
        self,
        modality_key_1: Union[str, Mapping[str, Any]] = "X_pca",
        modality_key_2: Union[str, Mapping[str, Any]] = "X_pca",
        **kwargs: Any,
    ) -> "IntegrationProblem[K]":
        """
        Prepare the :class:`moscot.problems.cross_modality.IntegrationProblem`.

        This method prepares the data to be passed to the optimal transport solver.

        Parameters
        ----------
        modality_key_1
            TODO
        modality_key_2
            TODO

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`.
        """

        x = {"attr": "obsm", "key": modality_key_1} if isinstance(modality_key_1, str) else modality_key_1
        y = {"attr": "obsm", "key": modality_key_2} if isinstance(modality_key_2, str) else modality_key_2

        self.adata.obs["dummy"] = 1  # TODO: find another solution
        self.adata_2.obs["dummy"] = 1  # TODO: find another solution

        return super().prepare(x=x, y=y, policy="sequential", key="dummy", **kwargs)

    @d.dedent
    def solve(
        self,
        epsilon: Optional[float] = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        cost: Literal["SqEuclidean"] = "SqEuclidean",
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
    ) -> "IntegrationProblem[K]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.cross_modality.IntegrationProblem`.

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

        Returns
        -------
        :class:`moscot.problems.cross_modality.IntegrationProblem`.
        """
        return super().solve(
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
        )  # type: ignore[return-value]

    @property
    def adata_2(self) -> AnnData:
        """Single-cell data."""
        return self._adata_2

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return (Policy.SEQUENTIAL, Policy.DUMMY)

    @property
    def _secondary_adata(self) -> Optional[AnnData]:
        return self._adata_2

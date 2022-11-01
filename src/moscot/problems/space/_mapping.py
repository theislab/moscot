from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional, Sequence

from anndata import AnnData

from moscot._types import ArrayLike, Str_Dict_t, ScaleCost_t, ProblemStage_t, QuadInitializer_t
from moscot._docs._docs import d
from moscot._constants._key import Key
from moscot.problems._utils import handle_joint_attr
from moscot._constants._constants import Policy
from moscot.problems.space._mixins import SpatialMappingMixin
from moscot.problems._subset_policy import DummyPolicy, ExternalStarPolicy
from moscot.problems.base._base_problem import OTProblem
from moscot.problems.base._compound_problem import B, K, CompoundProblem

__all__ = ["MappingProblem"]


@d.dedent
class MappingProblem(CompoundProblem[K, OTProblem], SpatialMappingMixin[K, OTProblem]):
    """
    Class for mapping single cell omics data onto spatial data, based on :cite:`nitzan:19`.

    The `MappingProblem` allows to match single cell and spatial omics data via optimal transport.

    Parameters
    ----------
    adata_sc
        Instance of :class:`anndata.AnnData` containing the single cell data.
    adata_sp
        Instance of :class:`anndata.AnnData` containing the spatial data.

    Examples
    --------
    See notebook TODO(@giovp) LINK NOTEBOOK for how to use it.
    """

    def __init__(self, adata_sc: AnnData, adata_sp: AnnData, **kwargs: Any):
        super().__init__(adata_sp, **kwargs)
        self._adata_sc = adata_sc
        # TODO(michalk8): rename to common_vars?
        self.filtered_vars: Optional[Sequence[str]] = None

    def _create_policy(  # type: ignore[override]
        self,
        policy: Literal["external_star"] = "external_star",
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[DummyPolicy, ExternalStarPolicy[K]]:
        """Private class to create DummyPolicy if no batches are present in the spatial anndata."""
        del policy
        if key is None:
            return DummyPolicy(self.adata, **kwargs)
        return ExternalStarPolicy(self.adata, key=key, **kwargs)

    def _create_problem(
        self,
        src: K,
        tgt: K,
        src_mask: ArrayLike,
        tgt_mask: ArrayLike,
        **kwargs: Any,
    ) -> OTProblem:
        return self._base_problem_type(
            adata=self.adata_sp,
            adata_tgt=self.adata_sc,
            src_obs_mask=src_mask,
            tgt_obs_mask=None,
            src_var_mask=self.filtered_vars,  # type: ignore[arg-type]
            tgt_var_mask=self.filtered_vars,  # type: ignore[arg-type]
            src_key=src,
            tgt_key=tgt,
            **kwargs,
        )

    @d.dedent
    def prepare(
        self,
        sc_attr: Str_Dict_t,
        batch_key: Optional[str] = None,
        spatial_key: Union[str, Mapping[str, Any]] = Key.obsm.spatial,
        var_names: Optional[Sequence[Any]] = None,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        cost: Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "MappingProblem[K]":
        """
        Prepare the :class:`moscot.problems.space.MappingProblem`.

        This method prepares the data to be passed to the optimal transport solver.

        Parameters
        ----------
        sc_attr
            Specifies the attributes of the single cell adata.

        %(batch_key)s
        %(spatial_key)s

        var_names
            List of shared features to be used for the linear problem. If None, it defaults to the intersection
            between ``adata_sc`` and ``adata_sp``. If an empty list is pass, it defines a quadratic problem.

        %(joint_attr)s
        %(cost)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`.
        """
        x = {"attr": "obsm", "key": spatial_key} if isinstance(spatial_key, str) else spatial_key
        y = {"attr": "obsm", "key": sc_attr} if isinstance(sc_attr, str) else sc_attr
        self.batch_key = batch_key
        if isinstance(spatial_key, str):
            self.spatial_key = spatial_key
        else:
            self.spatial_key = spatial_key["key"]
        self.filtered_vars = var_names
        if self.filtered_vars is not None:
            xy, kwargs = handle_joint_attr(joint_attr, kwargs)
            kwargs["xy"] = xy
        return super().prepare(x=x, y=y, policy="external_star", key=batch_key, cost=cost, a=a, b=b, **kwargs)

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-2,
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
    ) -> "MappingProblem[K]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.MappingProblem`.

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
        :class:`moscot.problems.space.MappingProblem`.
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
        )  # type: ignore[return-value]

    @property
    def adata_sc(self) -> AnnData:
        """Single-cell data."""
        return self._adata_sc

    @property
    def adata_sp(self) -> AnnData:
        """Spatial data, alias for :attr:`adata`."""
        return self.adata

    @property
    def filtered_vars(self) -> Optional[Sequence[str]]:
        """Filtered variables."""  # noqa: D401
        return self._filtered_vars

    @filtered_vars.setter
    def filtered_vars(self, value: Optional[Sequence[str]]) -> None:
        self._filtered_vars = self._filter_vars(var_names=value)

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return (Policy.EXTERNAL_STAR, Policy.DUMMY)

    @property
    def _secondary_adata(self) -> Optional[AnnData]:
        return self._adata_sc

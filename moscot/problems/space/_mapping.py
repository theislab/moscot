from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional, Sequence

from anndata import AnnData

from moscot._types import ArrayLike, Str_Dict_t, QuadInitializer_t
from moscot._docs._docs import d
from moscot._constants._key import Key
from moscot._constants._constants import Policy, ScaleCost
from moscot.problems.space._mixins import SpatialMappingMixin
from moscot.problems._subset_policy import DummyPolicy, ExternalStarPolicy
from moscot.problems.base._base_problem import OTProblem, ScaleCost_t, ProblemStage
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
        policy: Literal[Policy.EXTERNAL_STAR] = Policy.EXTERNAL_STAR,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[DummyPolicy, ExternalStarPolicy[K]]:
        """Private class to create DummyPolicy if no batches are present n the spatial anndata."""
        if key is None:
            return DummyPolicy(self.adata, **kwargs)
        return ExternalStarPolicy(self.adata, key=key, **kwargs)

    def _create_problem(
        self,
        src_mask: ArrayLike,
        tgt_mask: ArrayLike,
        **kwargs: Any,
    ) -> B:  # type: ignore[type-var]
        """Private class to mask anndatas."""
        adata_sp = self._mask(src_mask)
        return self._base_problem_type(  # type: ignore[return-value]
            adata_sp[:, self.filtered_vars] if self.filtered_vars is not None else adata_sp,
            self.adata_sc[:, self.filtered_vars] if self.filtered_vars is not None else self.adata_sc,
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

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`.
        """
        x = {"attr": "obsm", "key": spatial_key} if isinstance(spatial_key, str) else spatial_key
        y = {"attr": "obsm", "key": sc_attr} if isinstance(sc_attr, str) else sc_attr
        self.batch_key = batch_key
        self.filtered_vars = var_names
        if self.filtered_vars is not None:
            if joint_attr is not None:
                kwargs["xy"] = joint_attr
            else:
                kwargs["callback"] = "local-pca"
                kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}
        return super().prepare(x=x, y=y, policy="external_star", key=batch_key, **kwargs)

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-3,
        scale_cost: ScaleCost_t = ScaleCost.MEAN,
        rank: int = -1,
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage, Tuple[ProblemStage, ...]] = (ProblemStage.PREPARED, ProblemStage.SOLVED),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "MappingProblem[K]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.MappingProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(scale_cost)s
        %(rank)s
        %(ott_jax_batch_size)s
        %(stage)s
        %(initializer_quad)s
        %(initializer_kwargs)s
        %(solve_kwargs)s

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`.
        """
        scale_cost = ScaleCost(scale_cost) if isinstance(scale_cost, ScaleCost) else scale_cost
        return super().solve(
            alpha=alpha,
            epsilon=epsilon,
            scale_cost=scale_cost,
            rank=rank,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            **kwargs,
        )  # type:ignore[return-value]

    @property
    def adata_sc(self) -> AnnData:
        """Return single cell adata."""
        return self._adata_sc

    @property
    def adata_sp(self) -> AnnData:
        """Return spatial adata. Alias for :attr:`adata`."""
        return self.adata

    @property
    def filtered_vars(self) -> Optional[Sequence[str]]:
        """Return filtered variables."""
        return self._filtered_vars

    @filtered_vars.setter
    def filtered_vars(self, value: Optional[Sequence[str]]) -> None:
        """Return filtered variables."""
        self._filtered_vars = self._filter_vars(var_names=value)

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return (Policy.EXTERNAL_STAR,)

    @property
    def _secondary_adata(self) -> Optional[AnnData]:
        return self._adata_sc

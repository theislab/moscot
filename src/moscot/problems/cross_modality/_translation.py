from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

from anndata import AnnData
import numpy as np

from moscot import _constants
from moscot._docs._docs import d
from moscot._types import (
    ArrayLike,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
    Str_Dict_t,
)
from moscot.base.problems.compound_problem import B, CompoundProblem, K
from moscot.base.problems.problem import OTProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.cross_modality._mixins import CrossModalityTranslationMixin
from moscot.utils.subset_policy import DummyPolicy, ExternalStarPolicy
from moscot.base.output import BaseSolverOutput

__all__ = ["TranslationProblem"]


@d.dedent
class TranslationProblem(CompoundProblem[K, OTProblem], CrossModalityTranslationMixin[K, OTProblem]):
    """
    Class for integrating single cell multiomics data, based on :cite:`demetci-scot:22`.

    Parameters
    ----------
    adata_src
        Instance of :class:`anndata.AnnData` containing the source data.
    adata_tgt
        Instance of :class:`anndata.AnnData` containing the target data.
    """

    def __init__(self, adata_src: AnnData, adata_tgt: AnnData, **kwargs: Any):
        super().__init__(adata_src, **kwargs)
        self._adata_tgt = adata_tgt
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
            adata=self.adata_src,
            adata_tgt=self.adata_tgt,
            src_obs_mask=src_mask,
            tgt_obs_mask=None,
            src_key=src,
            tgt_key=tgt,
            **kwargs,
        )

    @d.dedent
    def prepare(
        self,
        src_attr: Str_Dict_t,
        tgt_attr: Str_Dict_t,
        var_names: Optional[Sequence[Any]] = None,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        cost: Union[
            Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"],
            Mapping[str, Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"]],
        ] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "TranslationProblem[K]":
        """
        Prepare the :class:`moscot.problems.cross_modality.TranslationProblem`.

        This method prepares the data to be passed to the optimal transport solver.

        Parameters
        ----------
        

        Returns
        -------
        :class:`moscot.problems.cross_modality.TranslationProblem`.

        """
        self._src_attr = src_attr if isinstance(src_attr, str) else src_attr['key']
        self._tgt_attr = tgt_attr if isinstance(tgt_attr, str) else tgt_attr['key']

        x = {"attr": "obsm", "key": src_attr} if isinstance(src_attr, str) else src_attr
        y = {"attr": "obsm", "key": tgt_attr} if isinstance(tgt_attr, str) else tgt_attr
        #self.filtered_vars = var_names
        #if self.filtered_vars is not None:
        #    xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        #else:
        #    xy = None
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, x, y = handle_cost(xy=xy, x=x, y=y, cost=cost)
        if xy is not None:
            kwargs["xy"] = xy
        return super().prepare(x=x, y=y, policy="external_star", key=None, cost=cost, a=a, b=b, **kwargs)

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
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        jit: bool = True,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        ranks: Union[int, Tuple[int, ...]] = -1,
        tolerances: Union[float, Tuple[float, ...]] = 1e-2,
        linear_solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "TranslationProblem[K]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.cross_modality.TranslationProblem`.

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
        :class:`moscot.problems.cross_modality.TranslationProblem`.

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
            gamma=gamma,
            gamma_rescale=gamma_rescale,
            ranks=ranks,
            tolerances=tolerances,
            linear_solver_kwargs=linear_solver_kwargs,
            device=device,
            **kwargs,
        )  # type: ignore[return-value]

    @property
    def adata_tgt(self) -> AnnData:
        """Target data."""
        return self._adata_tgt

    @property
    def adata_src(self) -> AnnData:
        """Source data."""
        return self.adata

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.EXTERNAL_STAR, _constants.DUMMY  # type: ignore[return-value]

    
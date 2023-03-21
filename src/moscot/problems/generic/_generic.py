from types import MappingProxyType
from typing import Any, Dict, List, Type, Tuple, Union, Literal, Mapping, Iterable, Optional

from anndata import AnnData

from moscot._types import ScaleCost_t, ProblemStage_t, QuadInitializer_t, SinkhornInitializer_t
from moscot._docs._docs import d
from moscot.problems.base import (  # type: ignore[attr-defined]
    OTProblem,
    CondOTProblem,
    CompoundProblem,
    NeuralOTProblem,
)
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot._constants._constants import Policy
from moscot.problems.generic._mixins import GenericAnalysisMixin
from moscot.problems.base._compound_problem import B, K

<<<<<<< HEAD
__all__ = ["SinkhornProblem", "GWProblem", "NeuralProblem", "ConditionalNeuralProblem"]
=======
__all__ = ["SinkhornProblem", "GWProblem", "FGWProblem", "NeuralProblem", "ConditionalNeuralProblem"]
>>>>>>> origin/conditional_not_precommit


@d.dedent
class SinkhornProblem(GenericAnalysisMixin[K, B], CompoundProblem[K, B]):
    """
    Class for solving linear OT problems.

    Parameters
    ----------
    %(adata)s
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        cost: Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "SinkhornProblem[K, B]":
        """
        Prepare the :class:`moscot.problems.generic.SinkhornProblem`.

        Parameters
        ----------
        %(key)s
        %(joint_attr)s
        %(policy)s
        %(cost_lin)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.generic.SinkhornProblem`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.

        Examples
        --------
        %(ex_prepare)s
        """
        self.batch_key = key
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, _, _ = handle_cost(xy=xy, cost=cost)
        return super().prepare(
            key=key,
            policy=policy,
            xy=xy,
            cost=cost,
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
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        jit: bool = True,
        threshold: float = 1e-3,
        lse_mode: bool = True,
        norm_error: int = 1,
        inner_iterations: int = 10,
        min_iterations: int = 0,
        max_iterations: int = 2000,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        cost_matrix_rank: Optional[int] = None,
        **kwargs: Any,
    ) -> "SinkhornProblem[K,B]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.generic.SinkhornProblem`.

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
        %(sinkhorn_lr_kwargs)s
        %(device_solve)s
        %(cost_matrix_rank)s
        %(kwargs_linear)s

        Returns
        -------
        :class:`moscot.problems.generic.SinkhornProblem`.

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
            gamma=gamma,
            gamma_rescale=gamma_rescale,
            cost_matrix_rank=cost_matrix_rank,
            device=device,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "explicit"


@d.get_sections(base="GWProblem", sections=["Parameters"])
@d.dedent
class GWProblem(GenericAnalysisMixin[K, B], CompoundProblem[K, B]):
    """
    Class for solving Gromov-Wasserstein problems.

    Parameters
    ----------
    %(adata)s
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        key: str,
        GW_x: Union[str, Mapping[str, Any]],
        GW_y: Union[str, Mapping[str, Any]],
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        cost: Union[
            Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"],
            Mapping[str, Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"]],
        ] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "GWProblem[K, B]":
        """
        Prepare the :class:`moscot.problems.generic.GWProblem`.

        Parameters
        ----------
        %(key)s
        %(GW_x)s
        %(GW_y)s
        %(joint_attr)s
        %(policy)s
        %(cost)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.generic.GWProblem`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.

        Examples
        --------
        %(ex_prepare)s
        """
        self.batch_key = key

        GW_updated: List[Dict[str, Any]] = [{}] * 2
        for i, z in enumerate([GW_x, GW_y]):
            if isinstance(z, str):
                GW_updated[i] = {"attr": "obsm", "key": z, "tag": "point_cloud"}  # cost handled by handle_cost
            elif isinstance(z, dict):
                GW_updated[i] = z
            else:
                raise TypeError("`GW_x` and `GW_y` must be of type `str` or `dict`.")

        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, x, y = handle_cost(xy=xy, x=GW_updated[0], y=GW_updated[1], cost=cost)
        return super().prepare(
            key=key,
            xy=xy,
            x=x,
            y=y,
            policy=policy,
            cost=cost,
            a=a,
            b=b,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        alpha: float = 1.0,
        epsilon: Optional[float] = 1e-3,
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
    ) -> "GWProblem[K,B]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.generic.GWProblem`.

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
        %(kwargs_quad)s

        Returns
        -------
        :class:`moscot.problems.generic.GWProblem`.

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
        )

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "explicit"


@d.dedent
class FGWProblem(GWProblem[K, B]):
    """
    Class for solving Fused Gromov-Wasserstein problems.

    Parameters
    ----------
    %(adata)s
    """

    @d.dedent
    def prepare(
        self,
        key: str,
        GW_x: Union[str, Mapping[str, Any]],
        GW_y: Union[str, Mapping[str, Any]],
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        cost: Union[
            Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"],
            Mapping[str, Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"]],
        ] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "FGWProblem[K, B]":
        """
        Prepare the :class:`moscot.problems.generic.FGWProblem`.

        Parameters
        ----------
        %(key)s
        %(GW_x)s
        %(GW_y)s
        %(joint_attr)s
        %(policy)s
        %(cost)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.generic.FGWProblem`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.

        Examples
        --------
        %(ex_prepare)s
        """
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        return super().prepare(key=key, GW_x=GW_x, GW_y=GW_y, xy=xy, policy=policy, cost=cost, a=a, b=b, **kwargs)

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
    ) -> "FGWProblem[K,B]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.generic.FGWProblem`.

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
        :class:`moscot.problems.generic.FGWProblem`.

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
        )


@d.dedent
class NeuralProblem(CompoundProblem[K, B], GenericAnalysisMixin[K, B]):
    """Class for solving Parameterized Monge Map problems / Neural OT problems."""

    @d.dedent
    def prepare(
        self,
        key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "NeuralProblem[K, B]":
        """Prepare the :class:`moscot.problems.generic.NeuralProblem[K, B]`."""
        self.batch_key = key
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        return super().prepare(
            key=key,
            policy=policy,
            xy=xy,
            a=a,
            b=b,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
        pos_weights: bool = False,
        dim_hidden: Iterable[int] = (64, 64, 64, 64),
        beta: float = 1.0,
        best_model_metric: Literal[
            "sinkhorn_forward", "sinkhorn"
        ] = "sinkhorn_forward",  # TODO(@MUCDK) include only backward sinkhorn
        iterations: int = 25000,  # TODO(@MUCDK): rename to max_iterations
        inner_iters: int = 10,
        valid_freq: int = 50,
        log_freq: int = 5,
        patience: int = 100,
        optimizer_f_kwargs: Dict[str, Any] = MappingProxyType({}),
        optimizer_g_kwargs: Dict[str, Any] = MappingProxyType({}),
        pretrain_iters: int = 15001,
        pretrain_scale: float = 3.0,
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        train_size: float = 1.0,
        **kwargs: Any,
    ) -> "NeuralProblem[K, B]":
        """Solve."""
        return super().solve(
            batch_size=batch_size,
            tau_a=tau_a,
            tau_b=tau_b,
            epsilon=epsilon,
            seed=seed,
            pos_weights=pos_weights,
            dim_hidden=dim_hidden,
            beta=beta,
            best_model_metric=best_model_metric,
            iterations=iterations,
            inner_iters=inner_iters,
            valid_freq=valid_freq,
            log_freq=log_freq,
            patience=patience,
            optimizer_f_kwargs=optimizer_f_kwargs,
            optimizer_g_kwargs=optimizer_g_kwargs,
            pretrain_iters=pretrain_iters,
            pretrain_scale=pretrain_scale,
            valid_sinkhorn_kwargs=valid_sinkhorn_kwargs,
            train_size=train_size,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type["NeuralProblem[K, B]"]:
        return NeuralOTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return Policy.SEQUENTIAL, Policy.TRIU, Policy.EXPLICIT


@d.dedent
class ConditionalNeuralProblem(CondOTProblem, GenericAnalysisMixin[K, B]):
    """Class for solving Conditional Parameterized Monge Map problems / Conditional Neural OT problems."""

    @d.dedent
    def prepare(
        self,
        key: str,
        joint_attr: str,
<<<<<<< HEAD
=======
        cond_dim: int,
>>>>>>> origin/conditional_not_precommit
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "ConditionalNeuralProblem[K, B]":
        """Prepare the :class:`moscot.problems.generic.ConditionalNeuralProblem`."""
        self.batch_key = key
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        return super().prepare(
            policy_key=key,
            policy=policy,
            xy=xy,
<<<<<<< HEAD
=======
            cond_dim=cond_dim,
>>>>>>> origin/conditional_not_precommit
            a=a,
            b=b,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
        pos_weights: bool = False,
        dim_hidden: Iterable[int] = (64, 64, 64, 64),
        beta: float = 1.0,
        best_model_metric: Literal[
            "sinkhorn_forward", "sinkhorn"
        ] = "sinkhorn_forward",  # TODO(@MUCDK) include only backward sinkhorn
        iterations: int = 25000,  # TODO(@MUCDK): rename to max_iterations
        inner_iters: int = 10,
        valid_freq: int = 50,
        log_freq: int = 5,
        patience: int = 100,
        optimizer_f_kwargs: Dict[str, Any] = MappingProxyType({}),
        optimizer_g_kwargs: Dict[str, Any] = MappingProxyType({}),
        pretrain_iters: int = 15001,
        pretrain_scale: float = 3.0,
<<<<<<< HEAD
=======
        combiner_kwargs: Dict[str, Any] = MappingProxyType({}),
>>>>>>> origin/conditional_not_precommit
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        train_size: float = 1.0,
        **kwargs: Any,
    ) -> "ConditionalNeuralProblem[K, B]":
        """Solve."""
        return super().solve(
            batch_size=batch_size,
            tau_a=tau_a,
            tau_b=tau_b,
            epsilon=epsilon,
            seed=seed,
            pos_weights=pos_weights,
            dim_hidden=dim_hidden,
            beta=beta,
            best_model_metric=best_model_metric,
            iterations=iterations,
            inner_iters=inner_iters,
            valid_freq=valid_freq,
            log_freq=log_freq,
            patience=patience,
            optimizer_f_kwargs=optimizer_f_kwargs,
            optimizer_g_kwargs=optimizer_g_kwargs,
            pretrain_iters=pretrain_iters,
            pretrain_scale=pretrain_scale,
<<<<<<< HEAD
=======
            combiner_kwargs=combiner_kwargs,
>>>>>>> origin/conditional_not_precommit
            valid_sinkhorn_kwargs=valid_sinkhorn_kwargs,
            train_size=train_size,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type["ConditionalNeuralProblem[K, B]"]:
        return CondOTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return Policy.SEQUENTIAL, Policy.TRIU, Policy.EXPLICIT

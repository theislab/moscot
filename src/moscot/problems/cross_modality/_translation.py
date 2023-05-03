import types
from typing import Any, Literal, Mapping, Optional, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._types import (
    ArrayLike,
    CostKwargs_t,
    OttCostFnMap_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
)
from moscot.base.problems.compound_problem import B, CompoundProblem, K
from moscot.base.problems.problem import OTProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.cross_modality._mixins import CrossModalityTranslationMixin
from moscot.utils.subset_policy import DummyPolicy, ExternalStarPolicy

__all__ = ["TranslationProblem"]


class TranslationProblem(CompoundProblem[K, OTProblem], CrossModalityTranslationMixin[K, OTProblem]):
    """Class for integrating single-cell multiomics data, based on :cite:`demetci-scot:22`.

    Parameters
    ----------
    adata_src
        Annotated data object containing the source distribution.
    adata_tgt
        Annotated data object containing the target distribution.
    kwargs
        Keyword arguments for :class:`~moscot.base.problems.compound_problem.BaseCompoundProblem`.
    """

    def __init__(self, adata_src: AnnData, adata_tgt: AnnData, **kwargs: Any):
        super().__init__(adata_src, **kwargs)
        self._adata_tgt = adata_tgt

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

    def prepare(
        self,
        src_attr: Union[str, Mapping[str, Any]],
        tgt_attr: Union[str, Mapping[str, Any]],
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        batch_key: Optional[str] = None,
        cost: OttCostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "TranslationProblem[K]":
        """Prepare the problem.

        Parameters
        ----------
        src_attr
            - If :class:`str`, it must refer to a key in :attr:`anndata.AnnData.obsm`.
            - If :class:`dict`, the dictionary stores `attr` (attribute of :class:`~anndata.AnnData`) and `key`
              (key of :class:`AnnData.{attr} <anndata.AnnData>`).
        tgt_attr
            - If :class:`str`, it must refer to a key in :attr:`anndata.AnnData.obsm`.
            - If :class:`dict`, the dictionary stores `attr` (attribute of :class:`~anndata.AnnData`) and `key`
              (key of :class:`AnnData.{attr} <anndata.AnnData>`).
        joint_attr
            - If `None`, the pure Gromov-Wasserstein case is computed.
            - If :class:`str`, it must refer to a key in :attr:`anndata.AnnData.obsm`.
            - If :class:`dict`, the dictionary stores `attr` (attribute of :class:`~anndata.AnnData`) and `key`
              (key of :class:`AnnData.{attr} <anndata.AnnData>`).
        batch_key
            If present, specify the batch key in :attr:`~anndata.AnnData.obs`.
        cost
            Cost between two points in dimension d. Only used if no precomputed cost matrix is passed.
            If `cost` is of type :obj:`str`, the cost will be used for all point clouds. If `cost` is of type
            :obj:`dict`, it is expected to have keys `x`, `y`, and/or `xy`, with values corresponding to the
            cost functions in the quadratic term of the source distribution, the quadratic term of the target
            distribution, and/or the linear term, respectively.
        a
            Specifies the left marginals. If of type :class:`str` the left marginals are taken from
            :attr:`~anndata.AnnData.obs` ``['{a}']``. If ``a`` is `None` uniform marginals are used.
        b
            Specifies the right marginals. If of type :class:`str` the right marginals are taken from
            :attr:`~anndata.AnnData.obs` ``['{b}']``. If `b` is `None` uniform marginals are used.
        kwargs
            Keyword arguments.

        Returns
        -------
        The prepared problem.
        """
        self._src_attr = {"attr": "obsm", "key": src_attr} if isinstance(src_attr, str) else src_attr
        self._tgt_attr = {"attr": "obsm", "key": tgt_attr} if isinstance(tgt_attr, str) else tgt_attr

        self.batch_key = batch_key
        x = {"attr": "obsm", "key": src_attr} if isinstance(src_attr, str) else src_attr
        y = {"attr": "obsm", "key": tgt_attr} if isinstance(tgt_attr, str) else tgt_attr
        if joint_attr is None:
            xy = {}  # type: ignore[var-annotated]
        else:
            xy, kwargs = handle_joint_attr(joint_attr, kwargs)
            joint_attr_1_shape = getattr(self.adata_src, xy["x_attr"])[xy["x_key"]].shape
            joint_attr_2_shape = getattr(self.adata_tgt, xy["y_attr"])[xy["y_key"]].shape
            if not joint_attr_1_shape[1] == joint_attr_2_shape[1]:
                raise ValueError("The `joint_attr` must be of same dimension.")
        xy, x, y = handle_cost(xy=xy, x=x, y=y, cost=cost, cost_kwargs=cost_kwargs)
        if xy:
            kwargs["xy"] = xy
        return super().prepare(x=x, y=y, policy="external_star", key=batch_key, cost=cost, a=a, b=b, **kwargs)

    def solve(  # type: ignore[override]
        self,
        alpha: Optional[float] = 1.0,
        epsilon: Optional[float] = 1e-2,
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
    ) -> "TranslationProblem[K]":
        """Solve the optimal transport problem.

        Parameters
        ----------
        alpha
            Interpolation parameter between quadratic term and linear term, between 0 and 1. `alpha=1` corresponds to
            pure Gromov-Wasserstein, while `alpha -> 0` corresponds to pure Sinkhorn.
        epsilon
            Entropic regularisation parameter.
        tau_a
            Unbalancedness parameter for left marginal between 0 and 1. `tau_a=1` means no unbalancedness
            in the source distribution. The limit of `tau_a` going to 0 ignores the left marginals.
        tau_b
            unbalancedness parameter for right marginal between 0 and 1. `tau_b=1` means no unbalancedness
            in the target distribution. The limit of `tau_b` going to 0 ignores the right marginals.
        rank
            Rank of solver. If `-1` standard / full-rank optimal transport is applied.
        scale_cost
            How to rescale the cost matrix. Implemented scalings are
            'median', 'mean', 'max_cost', 'max_norm' and 'max_bound'.
            Alternatively, a float factor can be given to rescale the cost such
            that ``cost_matrix /= scale_cost``.
        batch_size
            Number of data points the matrix-vector products are applied to at the same time. The larger, the more
            memory is required. Only used if no precomputed cost matrix is used.
        stage
            Stages of subproblems which are to be solved.
        initializer
            Initializer to use for the problem.
            If not low rank, the standard initializer is used (outer product of marginals).
            If low rank, available options are:

                - `random`
                - `rank2` :cite:`scetbon:21a`
                - `k-means` :cite:`scetbon:22b`
                - `generalized-k-means` :cite:`scetbon:22b`:

            If `None`, the low-rank initializer will be selected based on how the data is passed.
            If the cost matrix is passed (instead of the data), the random initializer is used,
            otherwise the K-means initializer.
        initializer_kwargs
            keyword arguments for the initializer.
        jit
            if True, automatically jits (just-in-time compiles) the function upon first call.
        min_iterations
            The minimum number of Sinkhorn iterations carried out before the error is computed and monitored.
        max_iterations
            The maximum number of Sinkhorn iterations.
        threshold
            If not `None`, set all entries below `threshold` to 0.
        linear_solver_kwargs
            Keyword arguments for the linear solver used in quadratic problems.
        device
            If not `None`, the output will be transferred to `device`.
        kwargs
            Keyword arguments to solve the underlying Optimal Transport problem.

        Returns
        -------
        The solved problem.

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

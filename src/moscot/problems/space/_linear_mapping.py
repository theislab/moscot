import types
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._types import (  # QuadInitializer_t,
    ArrayLike,
    CostKwargs_t,
    OttCostFn_t,
    Policy_t,
    ProblemStage_t,
    ScaleCost_t,
    SinkhornInitializer_t,
)
from moscot.base.problems.compound_problem import B, CompoundProblem, K
from moscot.base.problems.problem import OTProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.space._mixins import SpatialMappingMixin
from moscot.utils.subset_policy import DummyPolicy, ExternalStarPolicy

__all__ = ["LinearMappingProblem"]


class LinearMappingProblem(SpatialMappingMixin[K, OTProblem], CompoundProblem[K, OTProblem]):
    """Class for mapping single cell omics data onto spatial data, based on :cite:`nitzan:19`.

    This class is designed for linear mapping in a spatially-informed latent space.

    Parameters
    ----------
    adata_sc
        Annotated data object containing the single-cell data.
    adata_sp
        Annotated data object containing the spatial data.
    """

    def __init__(self, adata_sc: AnnData, adata_sp: AnnData):
        super().__init__(adata_sp)
        self._adata_sc = adata_sc
        # TODO(michalk8): rename to common_vars?
        self.filtered_vars: Optional[Sequence[str]] = None

    def _create_policy(
        self,
        policy: Literal["external_star"] = "external_star",
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[DummyPolicy, ExternalStarPolicy[K]]:
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

    def prepare(
        self,
        batch_key: Optional[str] = None,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        cost: OttCostFn_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "LinearMappingProblem[K]":
        """Prepare the linear mapping problem.

        Parameters
        ----------
        batch_key
            Key in :attr:`~anndata.AnnData.obs` where the slices are stored.
        joint_attr
            How to get the data that defines the :term:`linear problem`:

            - :obj:`None` - `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
              on :attr:`~anndata.AnnData.X` is computed.
            - :class:`str` - key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        cost
            Cost function to use. Valid options are:

            - :class:`str` - name of the cost function, see :func:`~moscot.costs.get_available_costs`.
            - :class:`dict` - a dictionary with the following keys and values:

              - ``'xy'`` - cost function for the :term:`linear term`, same as above.
        cost_kwargs
            Keyword arguments for the :class:`~moscot.base.cost.BaseCost` or any backend-specific cost.
        a
            Source :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the source marginals are stored.
            - :class:`bool` - if :obj:`True`,
              :meth:`estimate the marginals <moscot.base.problems.OTProblem.estimate_marginals>`,
              otherwise use uniform marginals.
            - :obj:`None` - uniform marginals.
        b
            Target :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the target marginals are stored.
            - :class:`bool` - if :obj:`True`,
              :meth:`estimate the marginals <moscot.base.problems.OTProblem.estimate_marginals>`,
              otherwise use uniform marginals.
            - :obj:`None` - uniform marginals.
        kwargs
            Keyword arguments for :meth:`~moscot.base.problems.CompoundProblem.prepare`.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`problems` - the prepared subproblems.
        - :attr:`solutions` - set to an empty :class:`dict`.
        - :attr:`batch_key` - key in :attr:`~anndata.AnnData.obs` where batches are stored.
        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - set to ``'linear'``.
        """
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, x, y = handle_cost(
            xy=xy, x=kwargs.pop("x", {}), y=kwargs.pop("y", {}), cost=cost, cost_kwargs=cost_kwargs, **kwargs
        )

        return super().prepare(  # type: ignore[return-value]
            xy=xy, x=x, y=y, policy="external_star", key=batch_key, cost=None, a=a, b=b, **kwargs
        )

    def solve(
        self,
        epsilon: float = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: SinkhornInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        threshold: float = 1e-3,
        lse_mode: bool = True,
        inner_iterations: int = 10,
        min_iterations: Optional[int] = None,
        max_iterations: Optional[int] = None,
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "LinearMappingProblem[K]":
        r"""Solve the mapping problem.

        .. seealso::
            - See :doc:`../notebooks/tutorials/400_spatial_mapping` on how to
              prepare and solve the :class:`~moscot.problems.space.MappingProblem`.

        Parameters
        ----------
        epsilon
            :term:`Entropic regularization`.
        tau_a
            Parameter in :math:`(0, 1]` that defines how much :term:`unbalanced <unbalanced OT problem>` is the problem
            on the source :term:`marginals`. If :math:`1`, the problem is :term:`balanced <balanced OT problem>`.
        tau_b
            Parameter in :math:`(0, 1]` that defines how much :term:`unbalanced <unbalanced OT problem>` is the problem
            on the target :term:`marginals`. If :math:`1`, the problem is :term:`balanced <balanced OT problem>`.
        rank
            Rank of the :term:`low-rank OT` solver :cite:`scetbon:21a`.
            If :math:`-1`, full-rank solver :cite:`cuturi:2013` is used.
        scale_cost
            How to re-scale the cost matrices. If a :class:`float`, the cost matrices
            will be re-scaled as :math:`\frac{\text{cost}}{\text{scale_cost}}`.
        batch_size
            Number of rows/columns of the cost matrix to materialize during the solver iterations.
            Larger value will require more memory.
        stage
            Stage by which to filter the :attr:`problems` to be solved.
        initializer
            How to initialize the solution. If :obj:`None`, ``'default'`` will be used for a full-rank solver and
            ``'rank2'`` for a low-rank solver.
        initializer_kwargs
            Keyword arguments for the ``initializer``.
        jit
            Whether to :func:`~jax.jit` the underlying :mod:`ott` solver.
        min_iterations
            Minimum number of :term:`Sinkhorn` iterations.
        max_iterations
            Maximum number of :term:`Sinkhorn` iterations.
        threshold
            Convergence threshold of the :term:`Sinkhorn` algorithm. In the :term:`balanced <balanced OT problem>` case,
            this is typically the deviation between the target :term:`marginals` and the marginals of the current
            :term:`transport matrix`. In the :term:`unbalanced <unbalanced OT problem>` case, the relative change
            between the successive solutions is checked.
        lse_mode
            Whether to use `log-sum-exp (LSE)
            <https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations>`_
            computations for numerical stability.
        inner_iterations
            Compute the convergence criterion every ``inner_iterations``.
        device
            Transfer the solution to a different device, see :meth:`~moscot.base.output.BaseSolverOutput.to`.
            If :obj:`None`, keep the output on the original device.
        kwargs
            Keyword arguments for :meth:`~moscot.base.problems.CompoundProblem.solve`.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`solutions` - the :term:`OT` solutions for each subproblem.
        - :attr:`stage` - set to ``'solved'``.
        """
        return super().solve(  # type: ignore[return-value]
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
            inner_iterations=inner_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            device=device,
            **kwargs,
        )

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
        """Filtered variables."""
        return self._filtered_vars

    @filtered_vars.setter
    def filtered_vars(self, value: Optional[Sequence[str]]) -> None:
        self._filtered_vars = self._filter_vars(var_names=value)  # type: ignore[misc]

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.EXTERNAL_STAR, _constants.DUMMY  # type: ignore[return-value]

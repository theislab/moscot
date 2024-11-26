import types
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._logging import logger
from moscot._types import (
    ArrayLike,
    CostKwargs_t,
    OttCostFnMap_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
    SinkhornInitializer_t,
)
from moscot.base.problems.compound_problem import B, Callback_t, CompoundProblem, K
from moscot.base.problems.problem import OTProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.space._mixins import SpatialMappingMixin
from moscot.utils.subset_policy import DummyPolicy, ExternalStarPolicy

__all__ = ["MappingProblem"]


class MappingProblem(SpatialMappingMixin[K, OTProblem], CompoundProblem[K, OTProblem]):
    """Class for mapping single cell omics data onto spatial data, based on :cite:`nitzan:19`.

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
        sc_attr: Optional[Union[str, Mapping[str, Any]]],
        batch_key: Optional[str] = None,
        spatial_key: Union[str, Mapping[str, Any]] = "spatial",
        var_names: Optional[Sequence[str]] = None,
        normalize_spatial: bool = True,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        cost: OttCostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        x_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        y_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        x_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        y_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        subset: Optional[Sequence[Tuple[K, K]]] = None,
        reference: Optional[Any] = None,
    ) -> "MappingProblem[K]":
        """Prepare the mapping problem problem.

        .. seealso::
            - See :doc:`../notebooks/tutorials/400_spatial_mapping` on how to
              prepare and solve the :class:`~moscot.problems.space.MappingProblem`.

        Parameters
        ----------
        sc_attr
            How to get the data for the :term:`quadratic term`. Usually, it’s the :attr:`~anndata.AnnData.X` attribute,
            which contains normalized counts, but a different modality or a pre-computed
            `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_ can also be used. Valid options are:

            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm`.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        batch_key
            Key in :attr:`~anndata.AnnData.obs` where the slices are stored.
        spatial_key
            Key in :attr:`~anndata.AnnData.obsm` where the spatial coordinates are stored.
        var_names
            Genes in :attr:`~anndata.AnnData.var_names` for the :term:`linear term` in the
            :term:`fused <fused Gromov-Wasserstein>` case. Valid options are:

            - :obj:`None` - use all genes shared between :attr:`adata_sp` and :attr:`adata_sc`.
            - :class:`~typing.Sequence` - use a subset of genes. If an empty sequence, the problem will correspond
              to the pure :term:`Gromov-Wasserstein` case.

            See also the ``joint_attribute`` parameter.
        normalize_spatial
            Whether to normalize the spatial coordinates. If :obj:`True`, the coordinates are normalized
            by standardizing them.
        joint_attr
            How to get the data for the :term:`linear term` in the :term:`fused <fused Gromov-Wasserstein>` case:

            - :obj:`None` - `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
              on :attr:`~anndata.AnnData.X` is computed.
            - :class:`str` - key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.
        cost
            Cost function to use. Valid options are:

            - :class:`str` - name of the cost function for all terms, see :func:`~moscot.costs.get_available_costs`.
            - :class:`dict` - a dictionary with the following keys and values:

              - ``'xy'`` - cost function for the :term:`linear term`.
              - ``'x'`` - cost function for the source :term:`quadratic term`.
              - ``'y'`` - cost function for the target :term:`quadratic term`.
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

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`problems` - the prepared subproblems.
        - :attr:`solutions` - set to an empty :class:`dict`.
        - :attr:`spatial_key` - key in :attr:`~anndata.AnnData.obsm` where the spatial coordinates are stored.
        - :attr:`batch_key` - key in :attr:`~anndata.AnnData.obs` where batches are stored.
        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - set to ``'quadratic'`` (if both `spatial_key` and `sc_attr` are passed)
            or ``'linear'`` (if both `spatial_key` and `sc_attr` are `None`).
        """
        if sc_attr:
            x = {"attr": "obsm", "key": spatial_key} if isinstance(spatial_key, str) else spatial_key
            y = {"attr": "obsm", "key": sc_attr} if isinstance(sc_attr, str) else sc_attr

            if normalize_spatial and x_callback is None:
                x_callback = "spatial-norm"
                if not len(x_callback_kwargs):
                    x_callback_kwargs = x
            if isinstance(x_callback, str) and x_callback in "spatial-norm":
                x = {}
            self.spatial_key = spatial_key if isinstance(spatial_key, str) else spatial_key["key"]
            logger.info("Preparing a :term:`quadratic problem`.")
        else:
            x = {}
            y = {}
            logger.info("Preparing a :term:`linear problem`.")
            if var_names and len(var_names) == 0:
                raise ValueError("Expected `var_names` to be non-empty for a :term:`linear problem`.")

        self.spatial_key = spatial_key if isinstance(spatial_key, str) else spatial_key["key"]
        self.batch_key = batch_key
        self.filtered_vars = var_names

        if self.filtered_vars is not None:
            xy, xy_callback, xy_callback_kwargs = handle_joint_attr(joint_attr, xy_callback, xy_callback_kwargs)
        else:
            xy = {}
        xy, x, y = handle_cost(
            xy=xy,
            x=x,
            y=y,
            cost=cost,
            cost_kwargs=cost_kwargs,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
        )
        return super().prepare(  # type: ignore[return-value]
            xy=xy,
            x=x,
            y=y,
            policy="external_star",
            key=batch_key,
            a=a,
            b=b,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
            x_callback_kwargs=x_callback_kwargs,
            y_callback_kwargs=y_callback_kwargs,
            xy_callback_kwargs=xy_callback_kwargs,
            subset=subset,
            reference=reference,
        )

    def solve(
        self,
        alpha: float = 0.5,
        epsilon: float = 1e-2,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: Union[QuadInitializer_t, SinkhornInitializer_t] = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        min_iterations: Optional[int] = None,
        max_iterations: Optional[int] = None,
        threshold: float = 1e-3,
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "MappingProblem[K]":
        r"""Solve the mapping problem.

        .. seealso::
            - See :doc:`../notebooks/tutorials/400_spatial_mapping` on how to
              prepare and solve the :class:`~moscot.problems.space.MappingProblem`.

        Parameters
        ----------
        alpha
            Parameter in :math:`(0, 1]` that interpolates between the :term:`quadratic term` and
            the :term:`linear term`. :math:`\alpha = 1` corresponds to the pure :term:`Gromov-Wasserstein` problem while
            :math:`\alpha \to 0` corresponds to the pure :term:`linear problem`.
        epsilon
            :term:`Entropic regularization`.
        tau_a
            Parameter in :math:`(0, 1]` that defines how much :term:`unbalanced <unbalanced OT problem>` is the problem
            on the source :term:`marginals`. If :math:`1`, the problem is :term:`balanced <balanced OT problem>`.
        tau_b
            Parameter in :math:`(0, 1]` that defines how much :term:`unbalanced <unbalanced OT problem>` is the problem
            on the target :term:`marginals`. If :math:`1`, the problem is :term:`balanced <balanced OT problem>`.
        rank
            Rank of the :term:`low-rank OT` solver :cite:`scetbon:21a,scetbon:21b`.
            If :math:`-1`, full-rank solver :cite:`cuturi:2013,peyre:2016` is used.
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
            Minimum number of :term:`(fused) GW <Gromov-Wasserstein>` or :term:`Sinkhorn` iterations,
            depending on `alpha`.
        max_iterations
            Maximum number of :term:`(fused) GW <Gromov-Wasserstein>` or :term:`Sinkhorn` iterations,
            depending on `alpha`.
        threshold
            Convergence threshold of the :term:`GW <Gromov-Wasserstein>` or the :term:`Sinkhorn` algorithm,
            depending on `alpha`.
        linear_solver_kwargs
            Keyword arguments for the inner :term:`linear problem` solver. Only used when `alpha` > 0.
        device
            Transfer the solution to a different device, see :meth:`~moscot.base.output.BaseDiscreteSolverOutput.to`.
            If :obj:`None`, keep the output on the original device.
        kwargs
            Keyword arguments for :meth:`~moscot.base.problems.CompoundProblem.solve`.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`solutions` - the :term:`OT` solutions for each subproblem.
        - :attr:`stage` - set to ``'solved'``.
        """
        additonal_kwargs = {}
        if self.problem_kind == "quadratic":
            additonal_kwargs["alpha"] = alpha
            additonal_kwargs["linear_solver_kwargs"] = linear_solver_kwargs
        else:
            if alpha != 0:
                raise ValueError("Expected `alpha` to be 0 for a `linear problem`.")
            additonal_kwargs.update(linear_solver_kwargs)

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
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            device=device,
            **kwargs,
            **additonal_kwargs,
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
        self._filtered_vars = self._filter_vars(var_names=value)

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.EXTERNAL_STAR, _constants.DUMMY  # type: ignore[return-value]

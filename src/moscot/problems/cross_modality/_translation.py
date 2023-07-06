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


class TranslationProblem(CrossModalityTranslationMixin[K, OTProblem], CompoundProblem[K, OTProblem]):
    """Class for integrating single-cell multi-omics data, based on :cite:`demetci-scot:22`.

    Parameters
    ----------
    adata_src
        Annotated data object containing the source modality.
    adata_tgt
        Annotated data object containing the target modality.
    kwargs
        Keyword arguments for :class:`~moscot.base.problems.CompoundProblem`.
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
        del tgt_mask
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
        a: Optional[Union[bool, str]] = None,
        b: Optional[Union[bool, str]] = None,
        **kwargs: Any,
    ) -> "TranslationProblem[K]":
        """Prepare the translation problem.

        .. seealso::
            - See :doc:`../notebooks/tutorials/600_tutorial_translation` on how to prepare the translation problem.

        Parameters
        ----------
        src_attr
            How to get the data for the source modality:

            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict`-  it should contain ``'attr'`` and ``'key'``, the attribute and the key
              in :class:`~anndata.AnnData`, and optionally ``'tag'``, one of :class:`~moscot.utils.tagged_array.Tag`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        tgt_attr
            How to get the data for the target modality:

            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict`-  it should contain ``'attr'`` and ``'key'``, the attribute and the key
              in :class:`~anndata.AnnData`, and optionally ``'tag'``, one of :class:`~moscot.utils.tagged_array.Tag`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        joint_attr
            How to get the data for the :term:`linear term` in the :term:`fused <fused Gromov-Wasserstein>` case:

            - :obj:`None` - the pure :term:`Gromov-Wasserstein` case is used.
            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict`-  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        batch_key
            Key in :attr:`~anndata.AnnData.obs` specifying the batch.
        cost
            Cost function to use. Valid options are:

            - :class:`str` - name of the cost function for all terms, see :func:`~moscot.costs.get_available_costs`.
            - :class:`dict` - a dictionary with the following keys and values:

              - ``'xy'`` - cost function for the :term:`linear term`.
              - ``'x'`` - cost function for the source modality.
              - ``'y'`` - cost function for the target modality.
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
        - :attr:`problem_kind` - set to ``'quadratic'``.
        """
        self._src_attr = {"attr": "obsm", "key": src_attr} if isinstance(src_attr, str) else src_attr
        self._tgt_attr = {"attr": "obsm", "key": tgt_attr} if isinstance(tgt_attr, str) else tgt_attr
        self.batch_key = batch_key

        if joint_attr is None:
            xy = {}  # type: ignore[var-annotated]
        else:
            xy, kwargs = handle_joint_attr(joint_attr, kwargs)
            _, dim_src = getattr(self.adata_src, xy["x_attr"])[xy["x_key"]].shape
            _, dim_tgt = getattr(self.adata_tgt, xy["y_attr"])[xy["y_key"]].shape
            if dim_src != dim_tgt:
                raise ValueError(
                    f"The dimensions of `joint_attr` do not match. "
                    f"The joint attribute in the source distribution has dimension {dim_src}, "
                    f"while the joint attribute in the target distribution has dimension {dim_tgt}."
                )
        xy, x, y = handle_cost(
            xy=xy, x=self._src_attr, y=self._tgt_attr, cost=cost, cost_kwargs=cost_kwargs  # type: ignore[arg-type]
        )
        if xy:
            kwargs["xy"] = xy
        return super().prepare(x=x, y=y, policy="external_star", key=batch_key, cost=cost, a=a, b=b, **kwargs)  # type: ignore[return-value] # noqa: E501

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
        r"""Solve the translation problem.

        .. seealso::
            - See :doc:`../notebooks/tutorials/600_tutorial_translation` on how to
              solve the :class:`~moscot.problems.cross_modality.TranslationProblem`.

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
            Rank of the :term:`low-rank OT` solver :cite:`scetbon:21b`.
            If :math:`-1`, full-rank solver :cite:`peyre:2016` is used.
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
            Minimum number of :term:`(fused) GW <Gromov-Wasserstein>` iterations.
        max_iterations
            Maximum number of :term:`(fused) GW <Gromov-Wasserstein>` iterations.
        threshold
            Convergence threshold of the :term:`GW <Gromov-Wasserstein>` solver.
        linear_solver_kwargs
            Keyword arguments for the inner :term:`linear problem` solver.
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
    def adata_src(self) -> AnnData:
        """Source data."""
        return self.adata

    @property
    def adata_tgt(self) -> AnnData:
        """Target data."""
        return self._adata_tgt

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.EXTERNAL_STAR, _constants.DUMMY  # type: ignore[return-value]

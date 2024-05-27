import types
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._types import (
    CostFnMap_t,
    CostKwargs_t,
    Numeric_t,
    OttCostFn_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
    SinkhornInitializer_t,
)
from moscot.base.problems.birth_death import BirthDeathMixin, BirthDeathProblem
from moscot.base.problems.compound_problem import B, Callback_t, CompoundProblem
from moscot.problems._utils import (
    handle_cost,
    handle_joint_attr,
    pop_callback_kwargs,
    pop_callbacks,
)
from moscot.problems.time._mixins import TemporalMixin

__all__ = ["TemporalProblem", "LineageProblem"]


class TemporalProblem(  # type: ignore[misc]
    TemporalMixin[Numeric_t, BirthDeathProblem], BirthDeathMixin, CompoundProblem[Numeric_t, BirthDeathProblem]
):
    """Class for analyzing time-series single cell data based on :cite:`schiebinger:19`.

    Parameters
    ----------
    adata
        Annotated data object.
    kwargs
        Keyword arguments for :class:`~moscot.base.problems.CompoundProblem`.
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    def prepare(
        self,
        time_key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "triu", "tril", "explicit"] = "sequential",
        cost: OttCostFn_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[Union[bool, str]] = None,
        b: Optional[Union[bool, str]] = None,
        marginal_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        subset: Optional[Sequence[Tuple[Numeric_t, Numeric_t]]] = None,
        reference: Optional[Any] = None,
        xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        x_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        y_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        x_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        y_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        x: Mapping[str, Any] = types.MappingProxyType({}),
        y: Mapping[str, Any] = types.MappingProxyType({}),
        xy: Mapping[str, Any] = types.MappingProxyType({}),
    ) -> "TemporalProblem":
        """Prepare the temporal problem problem.

        .. seealso::
            - See :doc:`../notebooks/tutorials/200_temporal_problem` on how to
              prepare and solve the :class:`~moscot.problems.time.TemporalProblem`.

        Parameters
        ----------
        time_key
            Key in :attr:`~anndata.AnnData.obs` where the time points are stored.
        joint_attr
            How to get the data that defines the :term:`linear problem`:

            - :obj:`None` - `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
              on :attr:`~anndata.AnnData.X` is computed.
            - :class:`str` - key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        policy
            Rule which defines how to construct the subproblems using :attr:`obs['{time_key}'] <anndata.AnnData.obs>`.
            Valid options are:

            - ``'sequential'`` - align subsequent time points ``[(t0, t1), (t1, t2), ...]``.
            - ``'triu'`` - upper triangular matrix ``[(t0, t1), (t0, t2), ..., (t1, t2), ...]``.
            - ``'tril'`` - lower triangular matrix ``[(t_n, t_n-1), (t_n, t0), ..., (t_n-1, t_n-2), ...]``.
            - ``'explicit'`` - explicit sequence of subsets passed via ``subset = [(b3, b0), ...]``.
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
              :meth:`estimate the marginals <moscot.base.problems.BirthDeathProblem.estimate_marginals>`,
              otherwise use uniform marginals.
            - :obj:`None` - set to :obj:`True` if :attr:`proliferation_key` or :attr:`apoptosis_key` is not :obj:`None`.
        b
            Target :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the target marginals are stored.
            - :class:`bool` - if :obj:`True`,
              :meth:`estimate the marginals <moscot.base.problems.BirthDeathProblem.estimate_marginals>`,
              otherwise use uniform marginals.
            - :obj:`None` - set to :obj:`True` if :attr:`proliferation_key` or :attr:`apoptosis_key` is not :obj:`None`.
        marginal_kwargs
            Keyword arguments for :meth:`~moscot.base.problems.BirthDeathProblem.estimate_marginals`.
            It always contains :attr:`proliferation_key` and :attr:`apoptosis_key`,
            see :meth:`score_genes_for_marginals` for more information.
        kwargs
            Keyword arguments for :meth:`~moscot.base.problems.CompoundProblem.prepare`.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`problems` - the prepared subproblems.
        - :attr:`solutions` - set to an empty :class:`dict`.
        - :attr:`temporal_key` - key in :attr:`~anndata.AnnData.obs` where time points are stored.
        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - set to ``'linear'``.
        """
        self.temporal_key = time_key

        callback_dict = {
            "x_callback": x_callback,
            "y_callback": y_callback,
            "xy_callback": xy_callback,
            "x_callback_kwargs": x_callback_kwargs,
            "y_callback_kwargs": y_callback_kwargs,
            "xy_callback_kwargs": xy_callback_kwargs,
        }
        callback_dict = {k: v for k, v in callback_dict.items() if v}
        del x_callback, y_callback, xy_callback, x_callback_kwargs, y_callback_kwargs, xy_callback_kwargs

        xy, callback_dict = handle_joint_attr(joint_attr, callback_dict)
        x_callback, y_callback, xy_callback = pop_callbacks(callback_dict)
        x_callback_kwargs, y_callback_kwargs, xy_callback_kwargs = pop_callback_kwargs(callback_dict)
        xy, x, y = handle_cost(
            xy=xy,
            x=x,
            y=y,
            cost=cost,
            cost_kwargs=cost_kwargs,
            xy_callback=xy_callback,
            x_callback=x_callback,
            y_callback=y_callback,
        )
        marginal_kwargs = dict(marginal_kwargs)
        if self.apoptosis_key is not None:
            marginal_kwargs["apoptosis_key"] = self.apoptosis_key
        if self.proliferation_key is not None:
            marginal_kwargs["proliferation_key"] = self.proliferation_key
        estimate_marginals = self.proliferation_key is not None or self.apoptosis_key is not None
        a = estimate_marginals if a is None else a
        b = estimate_marginals if b is None else b
        assert callback_dict == {}, f"Unknown callback arguments: {callback_dict}."

        return super().prepare(  # type: ignore[return-value]
            key=time_key,
            xy=xy,
            x=x,
            y=y,
            policy=policy,
            marginal_kwargs=marginal_kwargs,
            a=a,
            b=b,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
            x_callback_kwargs=x_callback_kwargs,
            y_callback_kwargs=y_callback_kwargs,
            xy_callback_kwargs=xy_callback_kwargs,
            reference=reference,
            subset=subset,
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
    ) -> "TemporalProblem":
        r"""Solve the temporal problem.

        .. seealso:
            - See :doc:`../notebooks/tutorials/200_temporal_problem` on how to
              prepare and solve the :class:`~moscot.problems.time.TemporalProblem`.

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
            How to re-scale the cost matrix. If a :class:`float`, the cost matrix
            will be re-scaled as :math:`\frac{\text{cost}}{\text{scale_cost}}`.
        batch_size
            Number of rows/columns of the cost matrix to materialize during the :term:`Sinkhorn` iterations.
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
        min_iterations
            Minimum number of :term:`Sinkhorn` iterations.
        max_iterations
            Maximum number of :term:`Sinkhorn` iterations.
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
        return super().solve(  # type:ignore[return-value]
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
    def _base_problem_type(self) -> Type[B]:
        return BirthDeathProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.TRIL, _constants.TRIU, _constants.EXPLICIT  # type: ignore[return-value] # noqa: E501


class LineageProblem(TemporalProblem):
    """Estimator for modelling time series single cell data based on :cite:`lange-moslin:23`.

    Parameters
    ----------
    adata
        Annotated data object.
    kwargs
        Keyword arguments for :class:`~moscot.problems.time.TemporalProblem`.
    """

    def prepare(
        self,
        time_key: str,
        lineage_attr: Mapping[str, Any] = types.MappingProxyType({}),
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "triu", "tril", "explicit"] = "sequential",
        cost: CostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        marginal_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        subset: Optional[Sequence[Tuple[Numeric_t, Numeric_t]]] = None,
        reference: Optional[Any] = None,
        xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        x_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        y_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
        xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        x_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        y_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        x: Mapping[str, Any] = types.MappingProxyType({}),
        y: Mapping[str, Any] = types.MappingProxyType({}),
        xy: Mapping[str, Any] = types.MappingProxyType({}),
    ) -> "LineageProblem":
        """Prepare the lineage problem problem.

        .. seealso::
            - See :doc:`../notebooks/tutorials/100_lineage` on how to
              prepare and solve the :class:`~moscot.problems.time.LineageProblem`.

        Parameters
        ----------
        time_key
            Key in :attr:`~anndata.AnnData.obs` where the time points are stored.
        lineage_attr
            How to get the lineage information, such as barcodes or lineage trees, for the :term:`quadratic term`:

            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.
              If an empty :class:`dict` is passed, use pre-computed cost matrices stored in
              :attr:`obsp['cost_matrices'] <anndata.AnnData.obsp>`.
        joint_attr
            How to get the data for the :term:`linear term` in the :term:`fused <fused Gromov-Wasserstein>` case:

            - :obj:`None` - `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
              on :attr:`~anndata.AnnData.X` is computed.
            - :class:`str` - key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        policy
            Rule which defines how to construct the subproblems using :attr:`obs['{time_key}'] <anndata.AnnData.obs>`.
            Valid options are:

            - ``'sequential'`` - align subsequent time points ``[(t0, t1), (t1, t2), ...]``.
            - ``'triu'`` - upper triangular matrix ``[(t0, t1), (t0, t2), ..., (t1, t2), ...]``.
            - ``'tril'`` - lower triangular matrix ``[(t_n, t_n-1), (t_n, t0), ..., (t_n-1, t_n-2), ...]``.
            - ``'explicit'`` - explicit sequence of subsets passed via ``subset = [(b3, b0), ...]``.
        cost
            Cost function to use. Valid options are:

            - :class:`str` - name of the cost function for all terms, see :func:`~moscot.costs.get_available_costs`.
            - :class:`dict` - a dictionary with the following keys and values:

              - ``'xy'`` - cost function for the :term:`linear term`.
              - ``'x'`` - cost function for the source :term:`quadratic term`, e.g., :func:`~moscot.costs.LeafDistance`
                or :func:`~moscot.costs.BarcodeDistance`.
              - ``'y'`` - cost function for the target :term:`quadratic term`, e.g., :func:`~moscot.costs.LeafDistance`
                or :func:`~moscot.costs.BarcodeDistance`.
        cost_kwargs
            Keyword arguments for the :class:`~moscot.base.cost.BaseCost` or any backend-specific cost.
        a
            Source :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the source marginals are stored.
            - :class:`bool` - if :obj:`True`,
              :meth:`estimate the marginals <moscot.base.problems.BirthDeathProblem.estimate_marginals>`,
              otherwise use uniform marginals.
            - :obj:`None` - set to :obj:`True` if :attr:`proliferation_key` or :attr:`apoptosis_key` is not :obj:`None`.
        b
            Target :term:`marginals`. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where the target marginals are stored.
            - :class:`bool` - if :obj:`True`,
              :meth:`estimate the marginals <moscot.base.problems.BirthDeathProblem.estimate_marginals>`,
              otherwise use uniform marginals.
            - :obj:`None` - set to :obj:`True` if :attr:`proliferation_key` or :attr:`apoptosis_key` is not :obj:`None`.
        marginal_kwargs
            Keyword arguments for :meth:`~moscot.base.problems.BirthDeathProblem.estimate_marginals`.
            It always contains :attr:`proliferation_key` and :attr:`apoptosis_key`,
            see :meth:`score_genes_for_marginals` for more information.
        kwargs
            Keyword arguments for :meth:`~moscot.base.problems.CompoundProblem.prepare`.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`problems` - the prepared subproblems.
        - :attr:`solutions` - set to an empty :class:`dict`.
        - :attr:`temporal_key` - key in :attr:`~anndata.AnnData.obs` where time points are stored.
        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - set to ``'quadratic'``.
        """
        if not len(lineage_attr) and ("cost_matrices" not in self.adata.obsp):
            raise KeyError("Unable to find cost matrices in `adata.obsp['cost_matrices']`.")

        x = y = lineage_attr

        callback_dict = {
            "x_callback": x_callback,
            "y_callback": y_callback,
            "xy_callback": xy_callback,
            "x_callback_kwargs": x_callback_kwargs,
            "y_callback_kwargs": y_callback_kwargs,
            "xy_callback_kwargs": xy_callback_kwargs,
        }
        callback_dict = {k: v for k, v in callback_dict.items() if v}
        del x_callback, y_callback, xy_callback, x_callback_kwargs, y_callback_kwargs, xy_callback_kwargs
        xy, callback_dict = handle_joint_attr(joint_attr, callback_dict)

        x_callback, y_callback, xy_callback = pop_callbacks(callback_dict)
        x_callback_kwargs, y_callback_kwargs, xy_callback_kwargs = pop_callback_kwargs(callback_dict)
        xy, x, y = handle_cost(
            xy=xy,
            x=x,
            y=y,
            cost=cost,  # type: ignore[arg-type]
            cost_kwargs=cost_kwargs,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
        )

        x.setdefault("attr", "obsp")
        x.setdefault("key", "cost_matrices")
        x.setdefault("cost", "custom")
        x.setdefault("tag", "cost_matrix")

        y.setdefault("attr", "obsp")
        y.setdefault("key", "cost_matrices")
        y.setdefault("cost", "custom")
        y.setdefault("tag", "cost_matrix")

        return super().prepare(  # type: ignore[return-value]
            time_key,
            joint_attr=xy,
            x=x,
            y=y,
            policy=policy,
            cost=cost,  # type: ignore[arg-type]
            a=a,
            b=b,
            marginal_kwargs=marginal_kwargs,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
            x_callback_kwargs=x_callback_kwargs,
            y_callback_kwargs=y_callback_kwargs,
            xy_callback_kwargs=xy_callback_kwargs,
            reference=reference,
            subset=subset,
        )

    def solve(
        self,
        alpha: float = 0.5,
        epsilon: float = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        min_iterations: Optional[int] = None,
        max_iterations: Optional[int] = None,
        threshold: float = 1e-3,
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "LineageProblem":
        r"""Solve the lineage problem.

        .. seealso::
            - See :doc:`../notebooks/tutorials/100_lineage` on how to
              prepare and solve the :class:`~moscot.problems.time.LineageProblem`.

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
            Keyword arguments for :meth:`~moscot.problems.time.TemporalProblem.solve`.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`solutions` - the :term:`OT` solutions for each subproblem.
        - :attr:`stage` - set to ``'solved'``.
        """
        return super().solve(  # type: ignore[return-value]
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
        )

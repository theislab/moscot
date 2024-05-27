import types
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._types import (
    CostKwargs_t,
    OttCostFn_t,
    OttCostFnMap_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
    SinkhornInitializer_t,
)
from moscot.base.problems.compound_problem import B, CompoundProblem, K
from moscot.base.problems.problem import OTProblem
from moscot.problems._utils import (
    handle_cost,
    handle_joint_attr,
    pop_callbacks_compound_prepare,
    pop_reference_subset,
)
from moscot.problems.generic._mixins import GenericAnalysisMixin

__all__ = ["SinkhornProblem", "GWProblem", "FGWProblem"]


def set_quad_defaults(z: Optional[Union[str, Mapping[str, Any]]]) -> Dict[str, str]:
    if isinstance(z, str):
        return {"attr": "obsm", "key": z, "tag": "point_cloud"}  # cost handled by handle_cost
    if isinstance(z, Mapping):
        return dict(z)
    raise TypeError("`x_attr` and `y_attr` must be of type `str` or `dict` if no callback is provided.")


class SinkhornProblem(GenericAnalysisMixin[K, B], CompoundProblem[K, B]):  # type: ignore[misc]
    """Class for solving a :term:`linear problem`.

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
        key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "explicit", "star"] = "sequential",
        cost: OttCostFn_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[Union[bool, str]] = None,
        b: Optional[Union[bool, str]] = None,
        **kwargs: Any,
    ) -> "SinkhornProblem[K, B]":
        r"""Prepare the individual :term:`linear subproblems <linear problem>`.

        .. seealso::
            - See :doc:`../notebooks/examples/problems/200_custom_cost_matrices` on how to pass custom cost matrices.
            - TODO(michalk8): add an example that shows how to pass different costs (with kwargs).

        Parameters
        ----------
        key
            Key in :attr:`~anndata.AnnData.obs` for the :class:`~moscot.utils.subset_policy.SubsetPolicy`.
        joint_attr
            How to get the data for the :term:`linear term`:

            - :obj:`None` - `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
              on :attr:`~anndata.AnnData.X` is computed.
            - :class:`str` - key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        policy
            Rule which defines how to construct the subproblems using :attr:`obs['{key}'] <anndata.AnnData.obs>`.
            Valid options are:

            - ``'sequential'`` - align subsequent categories.
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
        - :attr:`stage` - set to ``'prepared'``.
        - :attr:`problem_kind` - set to ``'linear'``.
        """
        self.batch_key = key  # type: ignore[misc]
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        quad_callback_keys = {"x", "y", "x_callback", "y_callback", "x_callback_kwargs", "y_callback_kwargs"}
        given_quad_keys = set(kwargs.keys()) & quad_callback_keys
        if given_quad_keys:
            raise ValueError(
                f"SinkhornProblem does not support passing: {given_quad_keys} since it is a linear problem."
            )
        reference, subset = pop_reference_subset(kwargs)  # TODO: make sure they are expected or not
        xy_callback, xy_callback_kwargs = kwargs.pop("xy_callback", None), kwargs.pop("xy_callback_kwargs", {})
        xy, _, _ = handle_cost(
            xy=xy,
            x={},
            y={},
            cost=cost,
            cost_kwargs=cost_kwargs,
            xy_callback=xy_callback,
        )
        if kwargs:
            raise ValueError(f"Unknown keyword arguments: {kwargs}.")
        return super().prepare(  # type: ignore[return-value]
            key=key,
            policy=policy,
            xy=xy,
            a=a,
            b=b,
            xy_callback=xy_callback,
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
    ) -> "SinkhornProblem[K,B]":
        r"""Solve the individual :term:`linear subproblems <linear problem>` \
        using the :term:`Sinkhorn` algorithm :cite:`cuturi:2013`.

        .. seealso:
            - See :doc:`../notebooks/examples/solvers/100_linear_problem_basic` on how to specify
              the most important parameters.
            - See :doc:`../notebooks/examples/solvers/200_linear_problems_advanced` on how to specify
              additional parameters, such as the ``initializer``.

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
        """  # noqa: D205
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
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.EXPLICIT, _constants.STAR  # type: ignore[return-value]


class GWProblem(GenericAnalysisMixin[K, B], CompoundProblem[K, B]):  # type: ignore[misc]
    """Class for solving the :term:`GW <Gromov-Wasserstein>` or :term:`FGW <fused Gromov-Wasserstein>` problems.

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
        key: str,
        x_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        y_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "explicit", "star"] = "sequential",
        cost: OttCostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[Union[bool, str]] = None,
        b: Optional[Union[bool, str]] = None,
        **kwargs: Any,
    ) -> "GWProblem[K, B]":
        """Prepare the individual :term:`quadratic subproblems <quadratic problem>`.

        .. seealso::
            - TODO(michalk8): add an example how to pass `x_attr/y_attr`.

        Parameters
        ----------
        key
            Key in :attr:`~anndata.AnnData.obs` for the :class:`~moscot.utils.subset_policy.SubsetPolicy`.
        x_attr
            How to get the data for the source :term:`quadratic term`:

            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.
            - :obj:`None` - ``'x_callback'`` must be passed via ``kwargs``.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        y_attr
            How to get the data for the target :term:`quadratic term`:

            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and the key
              in :class:`~anndata.AnnData`, and optionally ``'tag'``, one of :class:`~moscot.utils.tagged_array.Tag`.
            - :obj:`None` - ``'y_callback'`` must be passed via ``kwargs``.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        policy
            Rule which defines how to construct the subproblems. Valid options are:

            - ``'sequential'`` - align subsequent categories in :attr:`obs['{key}'] <anndata.AnnData.obs>`.
            - ``'explicit'`` - explicit sequence of subsets passed via ``subset = [(b3, b0), ...]``.
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
        self.batch_key = key  # type: ignore[misc]
        x = set_quad_defaults(x_attr) if "x_callback" not in kwargs else {}
        y = set_quad_defaults(y_attr) if "y_callback" not in kwargs else {}
        if "xy" in kwargs:
            raise ValueError("GWProblem does not support passing `xy`.")
        (
            x_callback,
            y_callback,
            xy_callback,
            xy_callback_kwargs,
            x_callback_kwargs,
            y_callback_kwargs,
            reference,
            subset,
        ) = pop_callbacks_compound_prepare(kwargs)

        xy, x, y = handle_cost(
            xy={},
            x=x,
            y=y,
            cost=cost,
            cost_kwargs=cost_kwargs,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
        )
        if kwargs:
            raise ValueError(f"Unknown keyword arguments: {kwargs}.")
        return super().prepare(  # type: ignore[return-value]
            key=key,
            xy=xy,
            x=x,
            y=y,
            policy=policy,
            a=a,
            b=b,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
            xy_callback_kwargs=xy_callback_kwargs,
            x_callback_kwargs=x_callback_kwargs,
            y_callback_kwargs=y_callback_kwargs,
            subset=subset,
            reference=reference,
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
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        min_iterations: Optional[int] = None,
        max_iterations: Optional[int] = None,
        threshold: float = 1e-3,
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "GWProblem[K,B]":
        r"""Solve the individual :term:`quadratic subproblems <quadratic problem>`.

        .. seealso:
            - See :doc:`../notebooks/examples/solvers/300_quad_problems_basic` on how to specify
              the most important parameters.
            - See :doc:`../notebooks/examples/solvers/400_quad_problems_advanced` on how to specify
              additional parameters, such as the ``initializer``.

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
        return super().solve(  # type: ignore[return-value]
            alpha=1.0,
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

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.EXPLICIT, _constants.STAR  # type: ignore[return-value]


class FGWProblem(GWProblem[K, B]):
    """Class for solving the :term:`FGW <fused Gromov-Wasserstein>` problem.

    Parameters
    ----------
    adata
        Annotated data object.
    kwargs
        Keyword arguments for :class:`~moscot.base.problems.CompoundProblem`.
    """

    def prepare(
        self,
        key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        x_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        y_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "explicit", "star"] = "sequential",
        cost: OttCostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[Union[bool, str]] = None,
        b: Optional[Union[bool, str]] = None,
        **kwargs: Any,
    ) -> "FGWProblem[K, B]":
        """Prepare the individual :term:`quadratic subproblems <quadratic problem>`.

        .. seealso::
            - TODO(michalk8): add an example how to pass `x_attr/y_attr`.

        Parameters
        ----------
        key
            Key in :attr:`~anndata.AnnData.obs` for the :class:`~moscot.utils.subset_policy.SubsetPolicy`.
        joint_attr
            How to get the data for the :term:`linear term` in the :term:`fused <fused Gromov-Wasserstein>` case:

            - :obj:`None` - run `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
              on :attr:`~anndata.AnnData.X` is computed.
            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and the key
              in :class:`~anndata.AnnData`, and optionally ``'tag'``, one of :class:`~moscot.utils.tagged_array.Tag`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        x_attr
            How to get the data for the source :term:`quadratic term`:

            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.
            - :obj:`None` - ``'x_callback'`` must be passed via ``kwargs``.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        y_attr
            How to get the data for the target :term:`quadratic term`:

            - :class:`str` - a key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict` -  it should contain ``'attr'`` and ``'key'``, the attribute and the key
              in :class:`~anndata.AnnData`, and optionally ``'tag'``, one of :class:`~moscot.utils.tagged_array.Tag`.
            - :obj:`None` - ``'y_callback'`` must be passed via ``kwargs``.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        policy
            Rule which defines how to construct the subproblems. Valid options are:

            - ``'sequential'`` - align subsequent categories in :attr:`obs['{key}'] <anndata.AnnData.obs>`.
            - ``'explicit'`` - explicit sequence of subsets passed via ``subset = [(b3, b0), ...]``.
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
        self.batch_key = key  # type: ignore[misc]
        x = (
            set_quad_defaults(x_attr) if "x_callback" not in kwargs else {}
        )  # does x_callback or set_default(x_attr) work?
        y = set_quad_defaults(y_attr) if "y_callback" not in kwargs else {}
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        (
            x_callback,
            y_callback,
            xy_callback,
            x_callback_kwargs,
            y_callback_kwargs,
            xy_callback_kwargs,
            reference,
            subset,
        ) = pop_callbacks_compound_prepare(kwargs)
        xy, x, y = handle_cost(
            xy=xy,
            x=x,
            y=y,
            cost=cost,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
            cost_kwargs=cost_kwargs,
        )
        if kwargs:
            raise ValueError(f"Unknown keyword arguments: {kwargs}.")
        return CompoundProblem.prepare(
            self,  # type: ignore[return-value, arg-type]
            key=key,
            xy=xy,
            x=x,
            y=y,
            policy=policy,
            a=a,
            b=b,
            reference=reference,
            subset=subset,
            x_callback=x_callback,
            y_callback=y_callback,
            xy_callback=xy_callback,
            x_callback_kwargs=x_callback_kwargs,
            y_callback_kwargs=y_callback_kwargs,
            xy_callback_kwargs=xy_callback_kwargs,
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
    ) -> "FGWProblem[K,B]":
        r"""Solve the individual :term:`quadratic subproblems <quadratic problem>`.

        .. seealso:
            - See :doc:`../notebooks/examples/solvers/300_quad_problems_basic` on how to specify
              the most important parameters.
            - See :doc:`../notebooks/examples/solvers/400_quad_problems_advanced` on how to specify
              additional parameters, such as the ``initializer``.

        Parameters
        ----------
        alpha
            Parameter in :math:`(0, 1)` that interpolates between the :term:`quadratic term` and
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
        if alpha == 1.0:
            raise ValueError("The `FGWProblem` is equivalent to the `GWProblem` when `alpha=1.0`.")
        return CompoundProblem.solve(
            self,  # type: ignore[return-value, arg-type]
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

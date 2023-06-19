import types
from typing import Any, Literal, Mapping, Optional, Tuple, Type, Union

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
from moscot.base.problems.compound_problem import B, CompoundProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.time._mixins import TemporalMixin

__all__ = ["TemporalProblem", "LineageProblem"]


class TemporalProblem(
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
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        cost: OttCostFn_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[Union[bool, str]] = None,
        b: Optional[Union[bool, str]] = None,
        marginal_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
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
            How to get the data defining the :term:`linear problem`:

            - :obj:`None` - `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
              on :attr:`~anndata.AnnData.X` is computed.
            - :class:`str` - key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict`-  it should contain ``'attr'`` and ``'key'``, the attribute and key in
              :class:`~anndata.AnnData`, and optionally ``'tag'`` from the
              :class:`tags <moscot.utils.tagged_array.Tag>`.

            By default, :attr:`tag = 'point_cloud' <moscot.utils.tagged_array.Tag.POINT_CLOUD>` is used.
        policy
            Rule which defines how to construct the subproblems using :attr:`obs['{time_key}'] <anndata.AnnData.obs>`.
            Valid options are:

            - ``'sequential'`` - align subsequent time points ``[(t0, t1), (t1, t2), ...]``.
            - ``'tril'`` - upper triangular matrix ``[(t0, t1), (t0, t2), ..., (t1, t2), ...]``.
            - ``'triu'`` - lower triangular matrix ``[(t_n, t_n-1), (t_n, t0), ..., (t_n-1, t_n-2), ...]``.
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
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, x, y = handle_cost(xy=xy, x=kwargs.pop("x", {}), y=kwargs.pop("y", {}), cost=cost, cost_kwargs=cost_kwargs)

        marginal_kwargs = dict(marginal_kwargs)
        marginal_kwargs["proliferation_key"] = self.proliferation_key
        marginal_kwargs["apoptosis_key"] = self.apoptosis_key

        estimate_marginals = self.proliferation_key is not None or self.apoptosis_key is not None
        a = estimate_marginals if a is None else a
        b = estimate_marginals if b is None else b

        return super().prepare(  # type: ignore[return-value]
            key=time_key,
            xy=xy,
            x=x,
            y=y,
            policy=policy,
            cost=None,  # cost information is already stored in x,y,xy
            marginal_kwargs=marginal_kwargs,
            a=a,
            b=b,
            **kwargs,
        )

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
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        threshold: float = 1e-3,
        lse_mode: bool = True,
        norm_error: int = 1,
        inner_iterations: int = 10,
        min_iterations: int = 0,
        max_iterations: int = 2000,
        cost_matrix_rank: Optional[int] = None,
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "TemporalProblem":
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
            norm_error=norm_error,
            inner_iterations=inner_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            cost_matrix_rank=cost_matrix_rank,
            device=device,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[B]:
        return BirthDeathProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.TRIU, _constants.EXPLICIT  # type: ignore[return-value]


class LineageProblem(TemporalProblem):
    """
    Estimator for modelling time series single cell data based on moslin.

    Class handling the computation and downstream analysis of temporal single cell data with lineage prior.

    Parameters
    ----------
    %(adata)s
    """

    def prepare(
        self,
        time_key: str,
        lineage_attr: Mapping[str, Any] = types.MappingProxyType({}),
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "sequential"] = "sequential",
        # TODO(michalk8): update
        cost: CostFnMap_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        a: Optional[str] = None,
        b: Optional[str] = None,
        marginal_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> "LineageProblem":
        """
        Prepare the :class:`moscot.problems.time.LineageProblem`.

        Parameters
        ----------
        %(time_key)s

        lineage_attr
            Specifies the way the lineage information is processed. TODO: Specify.

        %(joint_attr)s
        %(policy)s
        %(cost)s
        %(cost_kwargs)s
        %(a_temporal)s
        %(b_temporal)s
        %(marginal_kwargs)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.time.LineageProblem`

        Examples
        --------
        %(ex_prepare)s
        """
        if not len(lineage_attr) and ("cost_matrices" not in self.adata.obsp):
            raise KeyError("Unable to find cost matrices in `adata.obsp['cost_matrices']`.")

        x = y = lineage_attr

        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, x, y = handle_cost(xy=xy, x=x, y=y, cost=cost, cost_kwargs=cost_kwargs)

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
            cost=cost,
            a=a,
            b=b,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

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
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        jit: bool = True,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "LineageProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.time.LineageProblem`.

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
        %(linear_solver_kwargs)s
        %(device_solve)s
        %(kwargs_quad_fused)s

        Returns
        -------
        :class:`moscot.problems.time.LineageProblem`

        Examples
        --------
        %(ex_solve_quadratic)s
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

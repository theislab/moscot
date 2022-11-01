from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional, TYPE_CHECKING

import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._types import Numeric_t, ScaleCost_t, ProblemStage_t, QuadInitializer_t, SinkhornInitializer_t
from moscot._docs._docs import d
from moscot.problems._utils import handle_joint_attr
from moscot.solvers._output import BaseSolverOutput
from moscot._constants._constants import Policy
from moscot.problems.time._mixins import TemporalMixin
from moscot.problems.base._birth_death import BirthDeathMixin, BirthDeathProblem
from moscot.problems.base._compound_problem import B, CompoundProblem


@d.dedent
class TemporalProblem(
    TemporalMixin[Numeric_t, BirthDeathProblem], BirthDeathMixin, CompoundProblem[Numeric_t, BirthDeathProblem]
):
    """
    Class for analysing time series single cell data based on :cite:`schiebinger:19`.

    The `TemporalProblem` allows to model and analyse time series single cell data by matching
    cells from previous time points to later time points via optimal transport.
    Based on the assumption that the considered cell modality is similar in consecutive time points
    probabilistic couplings are computed between different time points.
    This allows to understand cell trajectories by inferring ancestors and descendants of single cells.

    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it.
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        time_key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        cost: Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Prepare the :class:`moscot.problems.time.TemporalProblem`.

        This method executes multiple steps to prepare the optimal transport problems.

        Parameters
        ----------
        %(time_key)s
        %(joint_attr)s
        %(policy)s
        %(cost)s
        %(a)s
        %(b)s
        %(kwargs_prepare)s


        Returns
        -------
        :class:`moscot.problems.time.TemporalProblem`.

        Raises
        ------
        KeyError
            If `time_key` is not in :attr:`anndata.AnnData.obs`.
        KeyError
            If `joint_attr` is a string and cannot be found in :attr:`anndata.AnnData.obsm`.

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.
        """
        self.temporal_key = time_key
        policy = Policy(policy)  # type: ignore[assignment]
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)

        # TODO(michalk8): needs to be modified
        marginal_kwargs = dict(kwargs.pop("marginal_kwargs", {}))
        marginal_kwargs["proliferation_key"] = self.proliferation_key
        marginal_kwargs["apoptosis_key"] = self.apoptosis_key
        if a is None:
            a = self.proliferation_key is not None or self.apoptosis_key is not None
        if b is None:
            b = self.proliferation_key is not None or self.apoptosis_key is not None

        return super().prepare(
            key=time_key,
            xy=xy,
            policy=policy,
            marginal_kwargs=marginal_kwargs,
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
        power: int = 1,
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: SinkhornInitializer_t = "default",
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
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.time.TemporalProblem`.

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
        %(kwargs_linear)s

        Returns
        -------
        :class:`moscot.problems.time.TemporalProblem`.
        """
        return super().solve(
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            rank=rank,
            scale_cost=scale_cost,
            power=power,
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
            device=device,
            **kwargs,
        )  # type:ignore[return-value]

    @property
    def prior_growth_rates(self) -> Optional[pd.DataFrame]:
        """Return the prior estimate of growth rates of the cells in the source distribution."""
        # TODO(michalk8): FIXME
        cols = ["prior_growth_rates"]
        df_list = [
            pd.DataFrame(problem.prior_growth_rates, index=problem.adata.obs.index, columns=cols)
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(
                    shape=(len(self.problems[tup].adata_tgt.obs), 1),
                    fill_value=np.nan,
                ),
                index=self.problems[tup].adata_tgt.obs.index,
                columns=cols,
            )
        )
        return pd.concat(df_list, verify_integrity=True)

    @property
    def posterior_growth_rates(self) -> Optional[pd.DataFrame]:
        """Return the posterior estimate of growth rates of the cells in the source distribution."""
        # TODO(michalk8): FIXME
        cols = ["posterior_growth_rates"]
        df_list = [
            pd.DataFrame(problem.posterior_growth_rates, index=problem.adata.obs.index, columns=cols)
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(
                    shape=(len(self.problems[tup].adata_tgt.obs), 1),
                    fill_value=np.nan,
                ),
                index=self.problems[tup].adata_tgt.obs.index,
                columns=cols,
            )
        )
        return pd.concat(df_list, verify_integrity=True)

    # TODO(michalk8): refactor me
    @property
    def cell_costs_source(self) -> Optional[pd.DataFrame]:
        """Return the cost of a cell obtained by the potentials of the optimal transport solution."""
        sol = list(self.problems.values())[0].solution
        if TYPE_CHECKING:
            assert isinstance(sol, BaseSolverOutput)
        if sol.potentials is None:
            return None
        df_list = [
            pd.DataFrame(
                np.array(np.abs(problem.solution.potentials[0])),  # type: ignore[union-attr,index]
                index=problem.adata_src.obs_names,
                columns=["cell_cost_source"],
            )
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(shape=(len(self.problems[tup].adata_tgt.obs), 1), fill_value=np.nan),
                index=self.problems[tup].adata_tgt.obs_names,
                columns=["cell_cost_source"],
            )
        )
        return pd.concat(df_list, verify_integrity=True)

    @property
    def cell_costs_target(self) -> Optional[pd.DataFrame]:
        """Return the cost of a cell (see online methods) obtained by the potentials of the OT solution."""
        sol = list(self.problems.values())[0].solution
        if TYPE_CHECKING:
            assert isinstance(sol, BaseSolverOutput)
        if sol.potentials is None:
            return None

        tup = list(self)[0]
        df_list = [
            pd.DataFrame(
                np.full(shape=(len(self.problems[tup].adata_src), 1), fill_value=np.nan),
                index=self.problems[tup].adata_src.obs_names,
                columns=["cell_cost_target"],
            )
        ]
        df_list.extend(
            [
                pd.DataFrame(
                    np.array(np.abs(problem.solution.potentials[1])),  # type: ignore[union-attr,index]
                    index=problem.adata_tgt.obs_names,
                    columns=["cell_cost_target"],
                )
                for problem in self.problems.values()
            ]
        )
        return pd.concat(df_list, verify_integrity=True)

    @property
    def _base_problem_type(self) -> Type[B]:
        return BirthDeathProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return Policy.SEQUENTIAL, Policy.TRIU, Policy.EXPLICIT


@d.dedent
class LineageProblem(TemporalProblem):
    """
    Estimator for modelling time series single cell data based on moslin.

    Class handling the computation and downstream analysis of temporal single cell data with lineage prior.

    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it.
    """

    @d.dedent
    def prepare(
        self,
        time_key: str,
        lineage_attr: Mapping[str, Any] = MappingProxyType({}),
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "sequential"] = "sequential",
        cost: Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
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
        %(a)s
        %(b)s
        %(kwargs_prepare)s

        Returns
        -------
        :class:`moscot.problems.time.LineageProblem`

        Raises
        ------
        KeyError
            If `time_key` is not in :attr:`anndata.AnnData.obs`.
        KeyError
            If `joint_attr` is a string and cannot be found in :attr:`anndata.AnnData.obsm`.
        ValueError
            If :attr:`adata.obsp` has no attribute `cost_matrices`.
        TypeError
            If `joint_attr` is not None, not a :class:`str` and not a :class:`dict`
        """
        if not len(lineage_attr) and ("cost_matrices" not in self.adata.obsp):
            raise KeyError("Unable to find cost matrices in `adata.obsp['cost_matrices']`.")

        lineage_attr = dict(lineage_attr)
        lineage_attr.setdefault("attr", "obsp")
        lineage_attr.setdefault("key", "cost_matrices")
        lineage_attr.setdefault("cost", "custom")
        lineage_attr.setdefault("tag", "cost")

        x = y = lineage_attr

        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        return super().prepare(
            time_key,
            joint_attr=xy,
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
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        power: int = 1,
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        jit: bool = True,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        warm_start: Optional[bool] = None,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        gw_unbalanced_correction: bool = True,
        ranks: Union[int, Tuple[int, ...]] = -1,
        tolerances: Union[float, Tuple[float, ...]] = 1e-2,
        linear_solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
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
        %(sinkhorn_lr_kwargs)s
        %(gw_lr_kwargs)s
        %(linear_solver_kwargs)s
        %(device_solve)s
        %(kwargs_quad_fused)s

        Returns
        -------
        :class:`moscot.problems.time.LineageProblem`
        """
        return super().solve(
            alpha=alpha,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            rank=rank,
            scale_cost=scale_cost,
            power=power,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            jit=jit,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            warm_start=warm_start,
            gamma=gamma,
            gamma_rescale=gamma_rescale,
            gw_unbalanced_correction=gw_unbalanced_correction,
            ranks=ranks,
            tolerances=tolerances,
            linear_solver_kwargs=linear_solver_kwargs,
            device=device,
            **kwargs,
        )

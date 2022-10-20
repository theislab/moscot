from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional, TYPE_CHECKING

import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._types import Numeric_t, ScaleCost_t, ProblemStage_t, QuadInitializer_t, SinkhornInitializer_t
from moscot._docs._docs import d
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
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Prepare the :class:`moscot.problems.time.TemporalProblem`.

        This method executes multiple steps to prepare the optimal transport problems.

        Parameters
        ----------
        %(time_key)s
        %(joint_attr)s
        %(joint_attr)s
        %(a)s
        %(b)s
        %(marginal_kwargs)s
        %(subset)s
        %(reference)s
        %(callback)s
        %(callback_kwargs)s


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
        if joint_attr is None:
            if "callback" not in kwargs:
                kwargs["callback"] = "local-pca"
            else:
                kwargs["callback"] = kwargs["callback"]
            kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}
        elif isinstance(joint_attr, str):
            kwargs["xy"] = {
                "x_attr": "obsm",
                "x_key": joint_attr,
                "y_attr": "obsm",
                "y_key": joint_attr,
            }
        elif isinstance(joint_attr, Mapping):
            kwargs["xy"] = joint_attr
        else:
            raise TypeError("TODO")

        # TODO(michalk8): needs to be modified
        marginal_kwargs = dict(marginal_kwargs)
        marginal_kwargs["proliferation_key"] = self.proliferation_key
        marginal_kwargs["apoptosis_key"] = self.apoptosis_key
        if "a" not in kwargs:
            kwargs["a"] = self.proliferation_key is not None or self.apoptosis_key is not None
        if "b" not in kwargs:
            kwargs["b"] = self.proliferation_key is not None or self.apoptosis_key is not None

        return super().prepare(
            key=time_key,
            policy=policy,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        epsilon: Optional[float] = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scale_cost: ScaleCost_t = "mean",
        rank: int = -1,
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
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.time.TemporalProblem`.

        Parameters
        ----------
        %(epsilon)s
        %(tau_a)s
        %(tau_b)s
        %(scale_cost)s
        %(rank)s
        %(ott_jax_batch_size)s
        %(stage)s
        %(initializer_lin)s
        %(initializer_kwargs)s
        %(jit)s
        %(sinkhorn_kwargs)s
        %(sinkhorn_lr_kwargs)s

        Returns
        -------
        :class:`moscot.problems.time.TemporalProblem`.
        """
        return super().solve(
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            scale_cost=scale_cost,
            rank=rank,
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
            **kwargs,
        )  # type:ignore[return-value]

    @property
    def growth_rates(self) -> Optional[pd.DataFrame]:
        """Growth rates of the cells estimated by posterior marginals."""
        # TODO: do we want to put this description above? (giovp) no, too long. Text for tutorial
        # If the OT problem is balanced, the posterior marginals
        # (approximately) equal the prior marginals (marginals defining the OT problem). In the unbalanced case the
        # marginals of the OT solution usually differ from the marginals of the original OT problem. This is an
        # indication of cell proliferation, i.e. a cell could have multiple descendants in the target distribution or
        # cell death, i.e. the cell is unlikely to have a descendant.
        # If multiple iterations are performed in :meth:`moscot.problems.time.TemporalProblem.solve` the number
        # of estimates for the cell growth rates equals is strictly larger than 2.
        # returns
        # TODO(michalk8): FIXME
        cols = ["growth_rates"]
        df_list = [
            pd.DataFrame(problem.growth_rates, index=problem.adata.obs.index, columns=cols)
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(
                    shape=(len(self.problems[tup]._adata_y.obs), 1),
                    fill_value=np.nan,
                ),
                index=self.problems[tup]._adata_y.obs.index,
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
        if sol.potentials[0] is None:
            return None
        df_list = [
            pd.DataFrame(
                np.array(problem.solution.potentials[0]), index=problem.adata.obs.index, columns=["cell_cost_source"]  # type: ignore[union-attr] # noqa: E501
            )
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(shape=(len(self.problems[tup]._adata_y.obs), 1), fill_value=np.nan),
                index=self.problems[tup]._adata_y.obs.index,
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
        if sol.potentials[0] is None:
            return None

        tup = list(self)[0]
        df_list = [
            pd.DataFrame(
                np.full(shape=(len(self.problems[tup].adata), 1), fill_value=np.nan),
                index=self.problems[tup].adata.obs.index,
                columns=["cell_cost_target"],
            )
        ]
        df_list.extend(
            [
                pd.DataFrame(
                    np.array(problem.solution.potentials[1]), index=problem._adata_y.obs.index, columns=["cell_cost_target"]  # type: ignore[union-attr] # noqa: E501
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
        %(marginal_kwargs)s
        %(a)s
        %(b)s
        %(subset)s
        %(reference)s
        %(callback)s
        %(callback_kwargs)s

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

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.
        """
        if not len(lineage_attr) and ("cost_matrices" not in self.adata.obsp):
            raise ValueError(
                "TODO: default location for quadratic loss is `adata.obsp[`cost_matrices`]` \
                        but adata has no key `cost_matrices` in `obsp`."
            )
        # TODO(michalk8): refactor me
        lineage_attr = dict(lineage_attr)
        lineage_attr.setdefault("attr", "obsp")
        lineage_attr.setdefault("key", "cost_matrices")
        lineage_attr.setdefault("loss", None)
        lineage_attr.setdefault("tag", "cost")
        lineage_attr.setdefault("loss_kwargs", {})
        x = y = lineage_attr

        return super().prepare(
            time_key,
            joint_attr=joint_attr,
            x=x,
            y=y,
            policy=policy,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scale_cost: ScaleCost_t = "mean",
        rank: int = -1,
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
    ) -> "LineageProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.time.LineageProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(tau_a)s
        %(tau_b)s
        %(scale_cost)s
        %(rank)s
        %(ott_jax_batch_size)s
        %(stage)s
        %(initializer_quad)s
        %(initializer_kwargs)s
        %(gw_kwargs)s
        %(sinkhorn_lr_kwargs)s

        Returns
        -------
        :class:`moscot.problems.time.LineageProblem`
        """
        return super().solve(
            alpha=alpha,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            scale_cost=scale_cost,
            rank=rank,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            jit=jit,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
        )

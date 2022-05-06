from types import MappingProxyType
from typing import Any, Dict, Type, Tuple, Union, Literal, Mapping, Optional
from numbers import Number

import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot._docs import d
from moscot.problems._compound_problem import B, SingleCompoundProblem
from moscot.analysis_mixins._time_analysis import TemporalAnalysisMixin
from moscot.problems.mixins._temporal_mixins import BirthDeathMixin, BirthDeathBaseProblem


@d.dedent
class TemporalProblem(SingleCompoundProblem[Number, BirthDeathBaseProblem], TemporalAnalysisMixin, BirthDeathMixin):
    """
    Class for analysing time series single cell data based on :cite:`schiebinger:19`.

    The `TemporalProblem` allows to model and analyse time series single cell data by matching
    cells from previous time points to later time points via optimal transport. Based on the
    assumption that the considered cell modality is similar in consecutive time points probabilistic
    couplings are computed between different time points.
    This allows to understand cell trajectories by inferring ancestors and descendants of single cells.

    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        time_key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Prepare the :class:`moscot.problems.time.TemporalProblem`.

        This method executes multiple steps to prepare the optimal transport problems.

        Parameters
        ----------
        time_key
            Key in :attr:`anndata.AnnData.obs` which defines the time point each cell belongs to. It is supposed
            to be of numerical data type.
        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps. If None, the data is read
            from :attr:`anndata.AnnData.X` and for each time point the corresponding PCA space is computed. If
            `joint_attr` is a string the data is assumed to be found in :attr:`anndata.AnnData.obsm`.
            If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of
            :attr:`anndata.AnnData` as a key and the corresponding attribute as a value.
        policy
            Defines which transport maps to compute given different cell distributions.
        %(marginal_kwargs)s
        subset
            Subset of `anndata.AnnData.obs` [key] values of which the policy is to be applied to.
        %(reference)s
        %(axis)s
        %(callback)s
        %(callback_kwargs)s
        kwargs
            Keyword arguments for :meth:`moscot.problems.CompoundBaseProblem._create_problems`.

        Returns
        -------
        :class:`moscot.problems.time.TemporalProblem`

        Raises
        ------
        KeyError
            If `time_key` is not in :attr:`anndata.AnnData.obs`.
        KeyError
            If `joint_attr` is a string and cannot be found in :attr:`anndata.AnnData.obsm`.
        """
        # TODO(michalk8): make a property + sanity checks?
        self._TEMPORAL_KEY = time_key

        if joint_attr is None:
            kwargs["callback"] = "local-pca"
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
    def push(
        self,
        start: Number,
        end: Number,
        result_key: Optional[str] = None,
        return_all: bool = False,
        scale_by_marginals: bool = True,
        **kwargs: Any,
    ) -> Optional[Union[npt.ArrayLike, Dict[Tuple[Any, Any], npt.ArrayLike]]]:
        """
        Push distribution of cells through time.

        Parameters
        ----------
        start
            Time point of source distribution.
        target
            Time point of target distribution.
        result_key
            Key of where to save the result in :attr:`anndata.AnnData.obs`. If None the result will be returned.
        return_all
            If `True` returns all the intermediate masses if pushed through multiple transport plans.
            If `True`, the result is returned as a dictionary.

        Returns
        -------
        Depending on `result_key` updates `adata` or returns the result. In the former case all intermediate results
        (corresponding to intermediate time points) are saved in :attr:`anndata.AnnData.obs`. In the latter case all
        intermediate step results are returned if `return_all` is `True`, otherwise only the distribution at `end`
        is returned.

        Raises
        ------
        %(CompoundBaseProblem_push.raises)s
        """
        if result_key is not None:
            return_all = True
        result = super().push(
            start=start,
            end=end,
            return_all=return_all,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )[start, end]

        if result_key is None:
            return result
        self._dict_to_adata(result, result_key)

    @d.dedent
    def pull(
        self,
        start: Number,
        end: Number,
        result_key: Optional[str] = None,
        return_all: bool = False,
        scale_by_marginals: bool = True,
        **kwargs: Any,
    ) -> Optional[Union[npt.ArrayLike, Dict[Tuple[Any, Any], npt.ArrayLike]]]:
        """
        Pull distribution of cells from time point `end` to time point `start`.

        Parameters
        ----------
        start
            Earlier time point, the time point the mass is pulled to.
        end
            Later time point, the time point the mass is pulled from.
        result_key
            Key of where to save the result in :attr:`anndata.AnnData.obs`. If `None` the result will be returned.
        return_all
            If `True` return all the intermediate masses if pushed through multiple transport plans. In this case the
            result is returned as a dictionary.

        Returns
        -------
        Depending on `result_key` updates `adata` or returns the result. In the former case all intermediate results
        (corresponding to intermediate time points) are saved in :attr:`anndata.AnnData.obs`. In the latter case all
        intermediate step results are returned if `return_all` is `True`, otherwise only the distribution at `start`
        is returned.

        Raises
        ------
        %(CompoundBaseProblem_pull.raises)s
        """
        if result_key is not None:
            return_all = True
        result = super().pull(
            start=start,
            end=end,
            return_all=return_all,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )[start, end]
        if result_key is None:
            return result
        self._dict_to_adata(result, result_key)

    @property
    def growth_rates(self) -> pd.DataFrame:
        """
        Growth rates of the cells estimated by posterior marginals.

        If the OT problem is balanced, the posterior marginals
        (approximately) equal the prior marginals (marginals defining the OT problem). In the unbalanced case the
        marginals of the OT solution usually differ from the marginals of the original OT problem. This is an
        indication of cell proliferation, i.e. a cell could have multiple descendants in the target distribution or
        cell death, i.e. the cell is unlikely to have a descendant.
        If multiple iterations are performed in :meth:`moscot.problems.time.TemporalProblem.solve` the number
        of estimates for the cell growth rates equals is strictly larger than 2.
        """
        cols = [f"g_{i}" for i in range(self.problems[list(self)[0]].growth_rates.shape[1])]
        df_list = [
            pd.DataFrame(problem.growth_rates, index=problem.adata.obs.index, columns=cols)
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(
                    shape=(len(self.problems[tup]._adata_y.obs), self.problems[tup].growth_rates.shape[1]),
                    fill_value=np.nan,
                ),
                index=self.problems[tup]._adata_y.obs.index,
                columns=cols,
            )
        )
        return pd.concat(df_list, verify_integrity=True)

    def _dict_to_adata(self, d: Mapping[str, npt.ArrayLike], obs_key: str) -> None:
        # TODO: np.full
        tmp = np.empty(len(self.adata))
        tmp[:] = np.nan
        for key, value in d.items():
            mask = self.adata.obs[self.temporal_key] == key
            tmp[mask] = np.squeeze(value)
        self.adata.obs[obs_key] = tmp

    @property
    def cell_costs_source(self) -> Optional[pd.DataFrame]:
        """
        Return the cost of a cell (see online methods) obtained by the potentials of the optimal transport solution.

        Raises
        ------
        NotImplementedError
            If the solver from :class:`moscot.solvers` does not use potentials
        """
        try:
            df_list = [
                pd.DataFrame(
                    problem.solution.potentials[0], index=problem.adata.obs.index, columns=["cell_cost_source"]
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
        except NotImplementedError:  # TODO(@MUCDK) check for specific error message
            return None

    @property
    def cell_costs_target(self) -> Optional[pd.DataFrame]:
        """Return the cost of a cell (see online methods) obtained by the potentials of the OT solution."""
        try:
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
                        problem.solution.potentials[1], index=problem._adata_y.obs.index, columns=["cell_cost_target"]
                    )
                    for problem in self.problems.values()
                ]
            )
            return pd.concat(df_list, verify_integrity=True)
        except NotImplementedError:  # TODO(@MUCDK) check for specific error message
            return None

    @property
    def _base_problem_type(self) -> Type[B]:
        return BirthDeathBaseProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "triu", "tril", "explicit"


class LineageProblem(TemporalProblem):
    """
    Estimator for modelling time series single cell data based on moslin.

    Class handling the computation and downstream analysis of temporal single cell data with lineage prior.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` instance containing the single cell data and corresponding metadata


    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it
    """

    def prepare(
        self,
        time_key: str,
        lineage_attr: Mapping[str, Any] = MappingProxyType({}),
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "LineageProblem":
        """
        Prepare the :class:`moscot.problems.time.LineageProblem`.

        This method executes multiple steps to prepare the problem for the Optimal Transport solver to be ready
        to solve it

        Parameters
        ----------
        time_key
            Key in :attr:`anndata.AnnData.obs` which defines the time point each cell belongs to. It is supposed to be
            of numerical data type.
        lineage_attr
            Specifies the way the lineage information is processed. TODO: Specify.
        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps. If None, the data is read
            from :attr:`anndata.AnnData.X` and for each time point the corresponding PCA space is computed.
            If `joint_attr` is a string the data is assumed to be found in :attr:`anndata.AnnData.obsm`.
            If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of
            :attr:`anndata.AnnData` as a key and the corresponding attribute as a value.
        policy
            defines which transport maps to compute given different cell distributions
        %(marginal_kwargs)s
        subset
            subset of `anndata.AnnData.obs` [key] values of which the policy is to be applied to
        %(reference)s
        %(axis)s
        %(callback)s
        %(callback_kwargs)s
        kwargs
            Keyword arguments for :meth:`moscot.problems.CompoundBaseProblem._create_problems`

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
        # TODO(michalk8): use and
        if not len(lineage_attr):
            if "cost_matrices" not in self.adata.obsp:
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
            x=x,
            y=y,
            policy=policy,
            **kwargs,
        )

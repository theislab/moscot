from tokenize import group
from typing import Any, Dict, Type, Tuple, Union, Literal, Optional, Sequence, Mapping
from numbers import Number

import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.problems import CompoundProblem, MultiMarginalProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._utils import beta, delta
from moscot.solvers._base_solver import BaseSolver
from moscot.mixins._time_analysis import TemporalAnalysisMixin
from moscot.problems._compound_problem import SingleCompoundProblem
from moscot.problems._base_problem import BaseProblem


class TemporalProblem(TemporalAnalysisMixin, SingleCompoundProblem):
    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None):
        self._temporal_key: Optional[str] = None
        super().__init__(adata, solver, base_problem_type=TemporalBaseProblem)

    def prepare(
        self,
        key: str,
        data_key: str = "X_pca",
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        subset: Optional[Sequence[Tuple[Any, Any]]] = None,
        a: Optional[Union[npt.ArrayLike, str]] = None,
        b: Optional[Union[npt.ArrayLike, str]] = None,
        estimate_marginals: bool = False,
        gene_set_proliferation: Sequence = None,
        gene_set_apoptosis: Sequence = None,
        proliferation_key: str = "_proliferation",
        apoptosis_key: str = "_apoptosis",
        reference: Optional[Any] = None,
        marginal_kwargs: Dict[Any, Any] = {},
        **kwargs: Any,
    ) -> "CompoundProblem":

        self._temporal_key = key
        if estimate_marginals:
            self._score_gene_set(gene_set_proliferation, proliferation_key, **kwargs)
            self._score_gene_set(gene_set_apoptosis, apoptosis_key, **kwargs)
            if a is not None or b is not None:
                raise Warning("The marginals are estimated by proliferation and apoptosis genes.")
            a = True
            b = True
            marginal_kwargs["proliferation_key"] = proliferation_key
            marginal_kwargs["apoptosis_key"] = apoptosis_key

        x = {"attr": "obsm", "key": data_key}
        y = {"attr": "obsm", "key": data_key}

        return super().prepare(
            key,
            policy,
            subset,
            reference,
            x=x,
            y=y,
            a=a,
            b=b,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    def _create_problems(self, init_kwargs: Dict[Any, Any] = {}, **kwargs: Any) -> Dict[Tuple[Any, Any], BaseProblem]:
        return {
            (x, y): self._base_problem_type(
                self._mask(x, x_mask, self._adata_src),
                self._mask(y, y_mask, self._adata_tgt),
                solver=self._solver,
                start=x,
                end=y,
                **init_kwargs
            ).prepare(**kwargs)
            for (x, y), (x_mask, y_mask) in self._policy.mask().items()
        }

    def push(self, 
            start: Any,
            end: Any,
            data: Optional[Union[str, npt.ArrayLike, Mapping[Tuple[Any, Any], Union[str, npt.ArrayLike]]]] = None,
            subset: Optional[Sequence[Any]] = None,
            normalize: bool = True,
            result_key: Optional[str] = None, 
            return_all: bool = False,
            return_as_dict: bool = False,
            scale_by_marginals: bool = True,
            **kwargs: Any,
            ) -> Union[npt.ArrayLike, Dict[Any, npt.ArrayLike]]:

        if result_key is None:
            return super().push(start=start, end=end, data=data, subset=subset, normalize=normalize, return_all=return_all, scale_by_marginals=scale_by_marginals, return_as_dict=return_as_dict, **kwargs)[start, end]
        result = super().push(start=start, end=end, data=data, subset=subset, normalize=normalize, return_all=True, scale_by_marginals=scale_by_marginals, return_as_dict=True, **kwargs)[start, end]
        self._dict_to_adata(result, result_key)

    def pull(self, 
            start: Any,
            end: Any,
            data: Optional[Union[str, npt.ArrayLike, Mapping[Tuple[Any, Any], Union[str, npt.ArrayLike]]]] = None,
            subset: Optional[Sequence[Any]] = None,
            normalize: bool = True,
            result_key: Optional[str] = None, 
            return_all: bool = False,
            return_as_dict: bool = False,
            scale_by_marginals: bool = True,
            **kwargs: Any,
            ) -> Union[npt.ArrayLike, Dict[Any, npt.ArrayLike]]:

        if result_key is None:
            return super().pull(start=start, end=end, data=data, subset=subset, normalize=normalize, return_all=return_all, scale_by_marginals=scale_by_marginals, return_as_dict=return_as_dict, **kwargs)[start, end]
        result = super().pull(start=start, end=end, data=data, subset=subset, normalize=normalize, return_all=True, scale_by_marginals=scale_by_marginals, return_as_dict=True, **kwargs)[start, end]
        self._dict_to_adata(result, result_key)

    def _dict_to_adata(self, d: Dict, key: str):
        tmp = np.empty(len(self.adata))
        tmp[:] = np.nan
        for key, value in d.items():
            mask = self.adata.obs[self._temporal_key] == key
            tmp[mask] = np.squeeze(value)
        self.adata.obs[key] = tmp

    def _score_gene_set(
        self, obs_str: str, gene_set: Optional[Sequence] = None, max_z_score: int = 5
    ):  # TODO(@MUCDK): Check whether we want to use sc.tl.score_genes() + add other options as in WOT
        if gene_set is None:
            self.adata.obs[obs_str] = 0
            return
        x = self.adata[:, gene_set]
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        std = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (x - mean) / std
        x[np.isnan(x)] = 0
        x[x < -max_z_score] = -max_z_score
        x[x > max_z_score] = max_z_score
        self.adata.obs[obs_str] = x

    def cell_type_transition(
        self,
        start: Any,
        end: Any,
        groups_key: str,
        early_cell_types: Optional[Sequence] = None,
        late_cell_types: Optional[Sequence] = None,
        forward=False,  # return value will be row-stochastic if forward=True, else column-stochastic
        **kwargs: Any
    ) -> pd.DataFrame:
        if late_cell_types is None:
            late_cell_types = list(self.adata.obs[groups_key].unique())
        if early_cell_types is None:
            early_cell_types = list(self.adata.obs[groups_key].unique())

        transition_table = pd.DataFrame(
            np.zeros((len(early_cell_types), len(late_cell_types))), index=early_cell_types, columns=late_cell_types
        )
        df_early = self.adata[self.adata.obs[self._temporal_key] == start].obs[[groups_key]].copy()
        df_late = self.adata[self.adata.obs[self._temporal_key] == end].obs[[groups_key]].copy()

        subsets = early_cell_types if forward else late_cell_types
        fun = self.push if forward else self.pull
        for subset in subsets:
            try:
                result = fun(start=start, end=end, data=groups_key, subset=subset, normalize=True, return_all=False, **kwargs)
                """result = self._apply(
                    data=key_groups,
                    subset=subset,
                    start=start,
                    end=end,
                    normalize=True,
                    return_all=False,
                    forward=forward,
                    scale_by_marginals=True,
                    return_as_dict=False,
                    **kwargs
                )[start, end]"""
            except ValueError: # this happens if there are requested cell types are not in the corresponding time step
                logging.info(f"No data points corresponding to {subset} found in `adata.obs[groups_key]` for {start if forward else end}")
                result = np.nan
            if forward:
                df_late.loc[:, "distribution"] = result
                target_cell_dist = (
                    df_late[df_late[groups_key].isin(late_cell_types)]#[[groups_key, "distribution"]]
                    .groupby(groups_key)
                    .sum()
                )
                transition_table.loc[subset, :] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else np.nan
                    for cell_type in late_cell_types
                ]
            else:
                df_early.loc[:, "distribution"] = result
                target_cell_dist = (
                    df_early[df_early[groups_key].isin(early_cell_types)]#[[groups_key, "distribution"]]
                    .groupby(groups_key)
                    .sum()
                )
                transition_table.loc[:, subset] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else np.nan
                    for cell_type in early_cell_types
                ]
        return transition_table

class TemporalBaseProblem(MultiMarginalProblem):
    def __init__(self, adata_x: AnnData, adata_y: AnnData, solver: Type[BaseSolver], start: Number, end: Number, **kwargs):
        
        assert start < end, f"{start} is expected to be strictly smaller than {end}"
        self._t_start = start
        self._t_end = end
        super().__init__(adata_x, adata_y=adata_y, solver=solver)

    def _estimate_marginals(
        self, adata: AnnData, source: bool, proliferation_key: str, apoptosis_key: str, **kwargs: Any
    ) -> npt.ArrayLike:
        if source:
            birth = beta(adata.obs[proliferation_key], **kwargs)
            death = delta(adata.obs[apoptosis_key], **kwargs)
            return np.exp((birth - death) * (self._t_end - self._t_start))
        return np.average(self._get_last_marginals[0])

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        self._a.append(np.asarray(sol.transport_matrix.sum(axis=1)))
        self._b.append(np.average(np.asarray(self._get_last_marginals()[0])))

    @property
    def growth_rates(self) -> npt.ArrayLike:
        return [np.power(self._a[i], 1 / (self._t_end - self._t_start)) for i in range(len(self._a))]

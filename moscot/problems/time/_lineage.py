from tokenize import group
from typing import Any, Dict, Type, Tuple, Union, Literal, Optional, Sequence, Mapping
from numbers import Number

import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
import numpy.typing as npt
import scanpy as sc
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
        self._proliferation_key: Optional[str] = None
        self._apoptosis_key: Optional[str] = None
        super().__init__(adata, solver, base_problem_type=TemporalBaseProblem)

    def score_genes_for_marginals(self,
                           gene_set_proliferation: Optional[Sequence] = None,
                           gene_set_apoptosis: Optional[Sequence] = None,
                           proliferation_key: str = "_proliferation",
                           apoptosis_key: str = "_apoptosis",
                           **kwargs: Any
                           ):
        self._proliferation_key = proliferation_key
        self._apoptosis_key = apoptosis_key
        if gene_set_proliferation is None:
            self.adata.obs[proliferation_key] = 0
        else:
            sc.tl.score_genes(self.adata, gene_set_proliferation, score_name=proliferation_key, **kwargs)
        if gene_set_apoptosis is None:
            self.adata.obs[apoptosis_key] = 0
        else:
            sc.tl.score_genes(self.adata, gene_set_apoptosis, score_name=apoptosis_key, **kwargs)

    def prepare(
        self,
        key: str,
        data_key: str = "X_pca",
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        subset: Optional[Sequence[Tuple[Any, Any]]] = None,
        a: Optional[Union[npt.ArrayLike, str]] = None,
        b: Optional[Union[npt.ArrayLike, str]] = None,
        reference: Optional[Any] = None,
        marginal_kwargs: Dict[Any, Any] = {},
        **kwargs: Any,
    ) -> "CompoundProblem":

        self._temporal_key = key
        if self._proliferation_key:
            if a is not None or b is not None:
                raise Warning("The marginals are estimated by proliferation and apoptosis genes.")
            a = True
            b = True
            marginal_kwargs["proliferation_key"] = self._proliferation_key
            marginal_kwargs["apoptosis_key"] = self._apoptosis_key

        x = {"attr": "obsm", "key": data_key}
        y = {"attr": "obsm", "key": data_key}

        return super().prepare(
            key=key,
            policy=policy,
            subset=subset,
            reference=reference,
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

    def cell_transition(
        self,
        start: Any,
        end: Any,
        early_cells: Union[str, Mapping[str, Sequence[Any]]],
        late_cells: Union[str, Mapping[str, Sequence[Any]]],
        forward=False,  # return value will be row-stochastic if forward=True, else column-stochastic
        **kwargs: Any
    ) -> pd.DataFrame:
        if isinstance(late_cells, str):
            _late_cells = list(self.adata.obs[late_cells].unique())
            _late_cells_key = late_cells
        else:
            _late_cells_key, _late_cells = late_cells.items()
            assert len(_late_cells_key) == 1, "The data can only be filtered according to one column of `adata.obs`"
            assert set(_late_cells).isin(set(self.adata.obs[_late_cells_key].unique())), f"Not all values {_late_cells} could be found in column {_late_cells_key}"
        if isinstance(early_cells, str):
            _early_cells = list(self.adata.obs[early_cells].unique())
            _early_cells_key = early_cells
        else:
            _early_cells_key, _early_cells = early_cells.items()
            assert len(_early_cells_key) == 1, "The data can only be filtered according to one column of `adata.obs`"
            assert set(_early_cells).isin(set(self.adata.obs[_early_cells_key].unique())), f"Not all values {_early_cells} could be found in column {_early_cells_key}"

        transition_table = pd.DataFrame(
            np.zeros((len(_early_cells), len(_late_cells))), index=_early_cells, columns=_late_cells
        )
        df_early = self.adata[self.adata.obs[self._temporal_key] == start].obs[[_early_cells_key]].copy()
        df_late = self.adata[self.adata.obs[self._temporal_key] == end].obs[[_late_cells_key]].copy()

        subsets = _early_cells if forward else _late_cells
        fun, key = self.push, _early_cells_key if forward else self.pull, _late_cells_key
        for subset in subsets:
            try:
                result = fun(start=start, end=end, data=key, subset=subset, normalize=True, return_all=False, **kwargs)
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
                    df_late[df_late[_late_cells_key].isin(_late_cells)]
                    .groupby(_late_cells_key)
                    .sum()
                )
                transition_table.loc[subset, :] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else np.nan
                    for cell_type in _late_cells
                ]
            else:
                df_early.loc[:, "distribution"] = result
                target_cell_dist = (
                    df_early[df_early[_early_cells_key].isin(_early_cells)]
                    .groupby(_early_cells_key)
                    .sum()
                )
                transition_table.loc[:, subset] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else np.nan
                    for cell_type in _early_cells
                ]
        return transition_table

class TemporalBaseProblem(MultiMarginalProblem):
    def __init__(self, adata_x: AnnData, adata_y: AnnData, start: Number, end: Number, solver: Optional[BaseSolver] = None, **kwargs):
        
        assert start < end, f"{start} is expected to be strictly smaller than {end}"
        self._t_start = start
        self._t_end = end
        super().__init__(adata_x, adata_y=adata_y, solver=solver, **kwargs)

    def _estimate_marginals(
        self, adata: AnnData, source: bool, proliferation_key: str, apoptosis_key: str, **kwargs: Any
    ) -> npt.ArrayLike:
        if source:
            birth = beta(adata.obs[proliferation_key], **kwargs)
            death = delta(adata.obs[apoptosis_key], **kwargs)
            return np.exp((birth - death) * (self._t_end - self._t_start))
        return np.average(self._get_last_marginals[0])

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        self._a.append(np.asarray(sol.a))
        self._b.append(np.average(sol.a)*np.ones(len(self._marginal_b_adata)))

    @property
    def growth_rates(self) -> npt.ArrayLike:
        return np.power(self.a, 1 / (self._t_end - self._t_start))

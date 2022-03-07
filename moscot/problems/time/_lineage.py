from types import MappingProxyType
from typing import Any, Dict, Tuple, Union, Literal, Mapping, Optional, Sequence
from numbers import Number
import logging

import pandas as pd
import scanpy as sc

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.problems import MultiMarginalProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._utils import beta, delta
from moscot.solvers._base_solver import BaseSolver
from moscot.mixins._time_analysis import TemporalAnalysisMixin
from moscot.problems._compound_problem import SingleCompoundProblem


class TemporalBaseProblem(MultiMarginalProblem):
    def __init__(
        self,
        adata_x: AnnData,
        adata_y: AnnData,
        start: Number,
        end: Number,
        solver: Optional[BaseSolver] = None,
        **kwargs: Any,
    ):
        super().__init__(adata_x, adata_y=adata_y, solver=solver, **kwargs)
        if start >= end:
            raise ValueError(f"{start} is expected to be strictly smaller than {end}")
        self._t_start = start
        self._t_end = end

    def _estimate_marginals(
        self,
        adata: AnnData,
        source: bool,
        proliferation_key: Optional[str] = None,
        apoptosis_key: Optional[str] = None,
        **kwargs: Any,
    ) -> npt.ArrayLike:
        if proliferation_key is None and apoptosis_key is None:
            raise ValueError("`proliferation_key` or `apooptosis_key` must be ")
        if proliferation_key is not None:
            birth = beta(adata.obs[proliferation_key], **kwargs)
        else:
            birth = 0
        if apoptosis_key is not None:
            death = delta(adata.obs[apoptosis_key], **kwargs)
        else:
            death = 0
        growth = np.exp((birth - death) * (self._t_end - self._t_start))
        if source:
            return growth
        return np.full(len(self._marginal_b_adata), np.average(growth))

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        _a = np.asarray(sol.a) / self._a[-1]
        self._a.append(_a)
        self._b.append(np.full(len(self._marginal_b_adata), np.average(_a)))

    @property
    def growth_rates(self) -> npt.ArrayLike:
        return np.transpose(np.power(self.a, 1 / (self._t_end - self._t_start)))


class TemporalProblem(TemporalAnalysisMixin, SingleCompoundProblem):
    _valid_policies = ["sequential", "pairwise", "triu", "tril", "explicit"]

    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None):
        super().__init__(adata, solver, base_problem_type=TemporalBaseProblem)
        self._temporal_key: Optional[str] = None
        self._proliferation_key: Optional[str] = None
        self._apoptosis_key: Optional[str] = None

    def score_genes_for_marginals(
        self,
        gene_set_proliferation: Optional[Sequence[str]] = None,
        gene_set_apoptosis: Optional[Sequence[str]] = None,
        proliferation_key: str = "proliferation",
        apoptosis_key: str = "apoptosis",
        **kwargs: Any,
    ) -> "TemporalProblem":
        if gene_set_proliferation is not None:
            sc.tl.score_genes(self.adata, gene_set_proliferation, score_name=proliferation_key, **kwargs)
            self._proliferation_key = proliferation_key
        if gene_set_apoptosis is not None:
            sc.tl.score_genes(self.adata, gene_set_apoptosis, score_name=apoptosis_key, **kwargs)
            self._apoptosis_key = apoptosis_key
        if gene_set_proliferation is None and gene_set_apoptosis is None:
            logging.info(
                "At least one of `gene_set_proliferation` or `gene_set_apoptosis` must be provided to score genes."
            )
        return self

    def prepare(
        self,
        key: str,
        data_key: str = "X_pca",
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        subset: Optional[Sequence[Tuple[Any, Any]]] = None,
        marginal_kwargs: Dict[str, Any] = {},  # we need this to be mutable
        **kwargs: Any,
    ) -> "TemporalProblem":
        if policy not in self._valid_policies:
            raise ValueError(f"The only valid policies for the {self.__str__} are {self._valid_policies}")

        x = {"attr": "obsm", "key": data_key}
        y = {"attr": "obsm", "key": data_key}
        self._temporal_key = key

        marginal_kwargs["proliferation_key"] = self._proliferation_key
        marginal_kwargs["apoptosis_key"] = self._apoptosis_key
        if "a" not in kwargs:
            kwargs["a"] = self._proliferation_key is not None or self._apoptosis_key is not None
        if "b" not in kwargs:
            kwargs["b"] = self._proliferation_key is not None or self._apoptosis_key is not None

        return super().prepare(
            key=key,
            policy=policy,
            subset=subset,
            x=x,
            y=y,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    def _create_problems(
        self, init_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any
    ) -> Dict[Tuple[Any, Any], TemporalBaseProblem]:
        return {
            (x, y): self._base_problem_type(
                self._mask(x, x_mask, self._adata_src),
                self._mask(y, y_mask, self._adata_tgt),
                solver=self._solver,
                start=x,
                end=y,
                **init_kwargs,
            ).prepare(**kwargs)
            for (x, y), (x_mask, y_mask) in self._policy.mask().items()
        }

    def push(
        self,
        start: Any,
        end: Any,
        result_key: Optional[str] = None,
        return_all: bool = False,
        scale_by_marginals: bool = True,
        **kwargs: Any,
    ) -> Optional[Union[npt.ArrayLike, Dict[Tuple[Any, Any], npt.ArrayLike]]]:
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

    def pull(
        self,
        start: Any,
        end: Any,
        result_key: Optional[str] = None,
        return_all: bool = False,
        scale_by_marginals: bool = True,
        **kwargs: Any,
    ) -> Optional[Union[npt.ArrayLike, Dict[Tuple[Any, Any], npt.ArrayLike]]]:
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

    def _dict_to_adata(self, d: Mapping[str, npt.ArrayLike], obs_key: str) -> None:
        tmp = np.empty(len(self.adata))
        tmp[:] = np.nan
        for key, value in d.items():
            mask = self.adata.obs[self._temporal_key] == key
            tmp[mask] = np.squeeze(value)
        self.adata.obs[obs_key] = tmp

    def _validate_args_cell_transition(
        self, arg: Union[str, Mapping[str, Sequence[Any]]]
    ) -> Tuple[Union[str, Sequence], Sequence]:
        if isinstance(arg, str):
            if not hasattr(self.adata.obs[arg], "cat"):
                raise ValueError(f"The column `{arg}` in `adata.obs` must be of categorical dtype")
            return arg, list(self.adata.obs[arg].unique())
        _key, _arg = arg.keys(), arg.values()
        if not hasattr(self.adata.obs[_key], "cat"):
            raise ValueError(f"The column `{_key}` in `adata.obs` must be of categorical dtype")
        assert len(_key) == 1, "The data can only be filtered according to one column of `adata.obs`"
        assert set(_arg).isin(
            set(self.adata.obs[_key].unique())
        ), f"Not all values {_arg} could be found in column {_key}"
        return _key, _arg

    def cell_transition(
        self,
        start: Any,
        end: Any,
        early_cells: Union[str, Mapping[str, Sequence[Any]]],
        late_cells: Union[str, Mapping[str, Sequence[Any]]],
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        **kwargs: Any,
    ) -> pd.DataFrame:
        _early_cells_key, _early_cells = self._validate_args_cell_transition(early_cells)
        _late_cells_key, _late_cells = self._validate_args_cell_transition(late_cells)

        transition_table = pd.DataFrame(
            np.zeros((len(_early_cells), len(_late_cells))), index=_early_cells, columns=_late_cells
        )

        df_late = self.adata[self.adata.obs[self._temporal_key] == end].obs[[_late_cells_key]].copy()
        df_early = self.adata[self.adata.obs[self._temporal_key] == start].obs[[_early_cells_key]].copy()

        if forward:
            _early_cells_present = set(_early_cells).intersection(set(df_early[_early_cells_key].unique()))
            for subset in _early_cells:
                if subset not in _early_cells_present:
                    transition_table.loc[subset, :] = np.nan
                    continue
                try:
                    result = self.push(
                        start=start,
                        end=end,
                        data=_early_cells_key,
                        subset=subset,
                        normalize=True,
                        return_all=False,
                        scale_by_marginals=True,
                        **kwargs,
                    )
                except ValueError as e:
                    if "no mass" in str(e):  # TODO: adapt
                        logging.info(
                            f"No data points corresponding to {subset} found in `adata.obs[groups_key]` for {start}"
                        )
                        result = np.nan
                    else:
                        raise
                df_late.loc[:, "distribution"] = result / np.sum(result)
                target_cell_dist = df_late[df_late[_late_cells_key].isin(_late_cells)].groupby(_late_cells_key).sum()
                transition_table.loc[subset, :] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else 0
                    for cell_type in _late_cells
                ]
            return transition_table
        _late_cells_present = set(_late_cells).intersection(set(df_late[_late_cells_key].unique()))
        for subset in _late_cells:
            if subset not in _late_cells_present:
                transition_table.loc[:, subset] = np.nan
                continue
            try:
                result = self.pull(
                    start=start,
                    end=end,
                    data=_late_cells_key,
                    subset=subset,
                    normalize=True,
                    return_all=False,
                    scale_by_marginals=True,
                    **kwargs,
                )
            except ValueError as e:
                if "no mass" in str(e):  # TODO: adapt
                    logging.info(f"No data points corresponding to {subset} found in `adata.obs[groups_key]` for {end}")
                    result = np.nan
                else:
                    raise
            df_early.loc[:, "distribution"] = result / np.sum(result)
            target_cell_dist = df_early[df_early[_early_cells_key].isin(_early_cells)].groupby(_early_cells_key).sum()
            transition_table.loc[:, subset] = [
                target_cell_dist.loc[cell_type, "distribution"]
                if cell_type in target_cell_dist.distribution.index
                else 0
                for cell_type in _early_cells
            ]
        return transition_table

    @property
    def growth_rates(self) -> pd.DataFrame:
        df = None
        for tup, problem in self._problems.items():
            if df is None:
                cols = [f"g_{i}" for i in range(problem.growth_rates.shape[1])]
                df = pd.DataFrame(columns=cols)
            df = df.append(
                pd.DataFrame(problem.growth_rates, index=self._problems[tup]._adata.obs.index, columns=cols),
                verify_integrity=True,
            )
        df = df.append(
            pd.DataFrame(
                np.full(
                    shape=(len(self._problems[tup]._adata_y.obs), problem.growth_rates.shape[1]), fill_value=np.nan
                ),
                index=self._problems[tup]._adata_y.obs.index,
                columns=cols,
            ),
            verify_integrity=True,
        )
        return df

    @property
    def proliferation_key(self) -> str:
        return self._proliferation_key

    @property
    def apoptosis_key(self) -> str:
        return self._apoptosis_key

    @proliferation_key.setter
    def proliferation_key(self, value):
        self._proliferation_key = value

    @apoptosis_key.setter
    def apoptosis_key(self, value):
        self._apoptosis_key = value

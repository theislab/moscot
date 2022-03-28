from types import MappingProxyType
from typing import Any, Dict, Tuple, Union, Literal, Mapping, Callable, Optional, Sequence
from numbers import Number
import logging

import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData
import scanpy as sc

from moscot.problems import MultiMarginalProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._utils import beta, delta
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.mixins._time_analysis import TemporalAnalysisMixin
from moscot.solvers._tagged_array import TaggedArray
from moscot.problems._compound_problem import SingleCompoundProblem

Callback_t = Optional[
    Union[
        Literal["pca_local"],
        Callable[[AnnData, Optional[AnnData], ProblemKind, Any], Tuple[TaggedArray, Optional[TaggedArray]]],
    ]
]


class TemporalBaseProblem(MultiMarginalProblem):
    def __init__(
        self,
        adata_x: AnnData,
        adata_y: AnnData,
        source: Number,
        target: Number,
        solver: Optional[BaseSolver] = None,
        **kwargs: Any,
    ):
        if source >= target:
            raise ValueError(f"{source} is expected to be strictly smaller than {target}.")
        super().__init__(adata_x, adata_y=adata_y, solver=solver, source=source, target=target, **kwargs)

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
        growth = np.exp((birth - death) * (self._target - self._source))
        if source:
            return growth
        return np.full(len(self._marginal_b_adata), np.average(growth))

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        _a = np.asarray(sol.a) / self._a[-1]
        self._a.append(_a)
        self._b.append(np.full(len(self._marginal_b_adata), np.average(_a)))

    @property
    def growth_rates(self) -> npt.ArrayLike:
        return np.transpose(np.power(self.a, 1 / (self._target - self._source)))


class TemporalProblem(TemporalAnalysisMixin, SingleCompoundProblem):
    _VALID_POLICIES = ["sequential", "pairwise", "triu", "tril", "explicit"]

    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None, **kwargs: Any):
        super().__init__(adata, solver=solver, base_problem_type=TemporalBaseProblem, **kwargs)
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
        if gene_set_proliferation is None:
            self._proliferation_key = None
        else:
            sc.tl.score_genes(self.adata, gene_set_proliferation, score_name=proliferation_key, **kwargs)
            self._proliferation_key = proliferation_key
        if gene_set_apoptosis is None:
            self._apoptosis_key = None
        else:
            sc.tl.score_genes(self.adata, gene_set_apoptosis, score_name=apoptosis_key, **kwargs)
            self._apoptosis_key = apoptosis_key
        if gene_set_proliferation is None and gene_set_apoptosis is None:
            logging.info(
                "At least one of `gene_set_proliferation` or `gene_set_apoptosis` must be provided to score genes."
            )
        return self

    def prepare(
        self,
        time_key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "TemporalProblem":
        if policy not in self._VALID_POLICIES:
            raise ValueError("TODO: wrong policies")
        self._temporal_key = time_key

        if joint_attr is None:
            kwargs["callback"] = "pca_local"
        elif isinstance(joint_attr, str):
            kwargs["x"] = kwargs["y"] = {"attr": "obsm", "key": joint_attr, "tag": "point_cloud"}  # TODO: pass loss
        elif not isinstance(joint_attr, dict):
            raise TypeError("TODO")

        marginal_kwargs = dict(marginal_kwargs)
        marginal_kwargs["proliferation_key"] = self._proliferation_key
        marginal_kwargs["apoptosis_key"] = self._apoptosis_key
        if "a" not in kwargs:
            kwargs["a"] = self._proliferation_key is not None or self._apoptosis_key is not None
        if "b" not in kwargs:
            kwargs["b"] = self._proliferation_key is not None or self._apoptosis_key is not None

        return super().prepare(
            key=time_key,
            policy=policy,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

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
        if len(_key) != 1:
            raise ValueError("The data can only be filtered according to one column of `adata.obs`")
        if not set(_arg).isin(set(self.adata.obs[_key].unique())):
            raise ValueError(f"Not all values {_arg} could be found in column {_key}")
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
        cols = [f"g_{i}" for i in range(list(self)[0][1].growth_rates.shape[1])]
        df_list = [
            pd.DataFrame(problem.growth_rates, index=self._problems[tup]._adata.obs.index, columns=cols)
            for tup, problem in self
        ]
        tup, problem = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(
                    shape=(len(self._problems[tup]._adata_y.obs), problem.growth_rates.shape[1]), fill_value=np.nan
                ),
                index=self._problems[tup]._adata_y.obs.index,
                columns=cols,
            ),
            verify_integrity=True,
        )
        return pd.concat(df_list)

    @property
    def proliferation_key(self) -> Optional[str]:
        return self._proliferation_key

    @property
    def apoptosis_key(self) -> Optional[str]:
        return self._apoptosis_key

    @proliferation_key.setter
    def proliferation_key(self, value: Optional[str] = None) -> None:
        self._proliferation_key = value

    @apoptosis_key.setter
    def apoptosis_key(self, value: Optional[str] = None) -> None:
        self._apoptosis_key = value


class LineageProblem(TemporalProblem):
    def prepare(
        self,
        time_key: str,
        lineage_attr: Mapping[str, Any] = MappingProxyType({}),
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "LineageProblem":
        if not len(lineage_attr):
            if "cost_matrices" not in self.adata.obsp:
                raise ValueError(
                    "TODO: default location for quadratic loss is `adata.obsp[`cost_matrices`]` \
                        but adata has no key `cost_matrices` in `obsp`."
                )
        lineage_attr = dict(lineage_attr)
        lineage_attr.setdefault("attr", "obsp")
        lineage_attr.setdefault("key", "cost_matrices")
        lineage_attr.setdefault("loss", None)
        lineage_attr.setdefault("tag", "cost")
        lineage_attr.setdefault("loss_kwargs", {})
        x = y = lineage_attr

        if joint_attr is None:
            kwargs["callback"] = "pca_local"
        elif isinstance(joint_attr, str):
            kwargs["joint_attr"] = {"attr": "obsm", "key": joint_attr, "tag": "point_cloud"}  # TODO: pass loss
        elif not isinstance(joint_attr, dict):
            raise TypeError("TODO")

        return super().prepare(
            time_key=time_key,
            policy=policy,
            x=x,
            y=y,
            **kwargs,
        )

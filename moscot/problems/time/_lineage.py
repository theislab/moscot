from types import MappingProxyType
from typing import Any, Dict, Type, Tuple, Union, Literal, Mapping, Optional, Sequence
from numbers import Number
import logging

import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData
import scanpy as sc

from moscot.problems import MultiMarginalProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._utils import beta, delta, MarkerGenes
from moscot.mixins._time_analysis import TemporalAnalysisMixin
from moscot.problems._compound_problem import B, SingleCompoundProblem


class TemporalBaseProblem(MultiMarginalProblem):
    def __init__(
        self,
        adata_x: AnnData,
        adata_y: AnnData,
        source: Number,
        target: Number,
    ):
        if source >= target:
            raise ValueError(f"{source} is expected to be strictly smaller than {target}.")
        super().__init__(adata_x, adata_y=adata_y, source=source, target=target)

    def _estimate_marginals(
        self,
        adata: AnnData,
        source: bool,
        proliferation_key: Optional[str] = None,
        apoptosis_key: Optional[str] = None,
        **kwargs: Any,
    ) -> npt.ArrayLike:
        if proliferation_key is None and apoptosis_key is None:
            raise ValueError("TODO: `proliferation_key` or `apoptosis_key` must be provided to estimate marginals")
        if proliferation_key is not None and proliferation_key not in adata.obs.columns:
            raise KeyError(f"TODO: {proliferation_key} not in `adata.obs`")
        if apoptosis_key is not None and apoptosis_key not in adata.obs.columns:
            raise KeyError(f"TODO: {apoptosis_key} not in `adata.obs`")
        if proliferation_key is not None:
            birth = beta(adata.obs[proliferation_key].to_numpy(), **kwargs)
        else:
            birth = 0
        if apoptosis_key is not None:
            death = delta(adata.obs[apoptosis_key].to_numpy(), **kwargs)
        else:
            death = 0
        growth = np.exp((birth - death) * (self._target - self._source))
        if source:
            return growth
        return np.full(len(self._adata_y), np.average(growth))

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        _a = np.asarray(sol.a) / self._a[-1]
        self._a.append(_a)
        # TODO(michalk8): sol._ones
        self._b.append(np.full(len(self._adata_y), np.average(_a)))

    @property
    def growth_rates(self) -> npt.ArrayLike:
        return np.power(self.a, 1 / (self._target - self._source))


class TemporalProblem(TemporalAnalysisMixin, SingleCompoundProblem[Number, TemporalBaseProblem]):
    def __init__(self, adata: AnnData):
        super().__init__(adata)
        self._temporal_key: Optional[str] = None
        self._proliferation_key: Optional[str] = None
        self._apoptosis_key: Optional[str] = None

    def score_genes_for_marginals(
        self,
        gene_set_proliferation: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        gene_set_apoptosis: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        proliferation_key: str = "proliferation",
        apoptosis_key: str = "apoptosis",
        **kwargs: Any,
    ) -> "TemporalProblem":
        # TODO(michalk8): make slightly more compact
        if gene_set_proliferation is None:
            self.proliferation_key = None
        else:
            if isinstance(gene_set_proliferation, str):
                sc.tl.score_genes(
                    self.adata,
                    MarkerGenes.proliferation_markers(gene_set_proliferation),
                    score_name=proliferation_key,
                    **kwargs,
                )
            else:
                sc.tl.score_genes(self.adata, gene_set_proliferation, score_name=proliferation_key, **kwargs)
            self.proliferation_key = proliferation_key
        if gene_set_apoptosis is None:
            self.apoptosis_key = None
        else:
            if isinstance(gene_set_apoptosis, str):
                sc.tl.score_genes(
                    self.adata,
                    MarkerGenes.apoptosis_markers(gene_set_apoptosis),
                    score_name=apoptosis_key,
                    **kwargs,
                )
            else:
                sc.tl.score_genes(self.adata, gene_set_apoptosis, score_name=apoptosis_key, **kwargs)
            self.apoptosis_key = apoptosis_key
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
        # TODO(michalk8): make a property + sanity checks
        self._temporal_key = time_key

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

    @property
    def growth_rates(self) -> pd.DataFrame:
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
        tmp = np.empty(len(self.adata))
        tmp[:] = np.nan
        for key, value in d.items():
            mask = self.adata.obs[self._temporal_key] == key
            tmp[mask] = np.squeeze(value)
        self.adata.obs[obs_key] = tmp

    @property
    def proliferation_key(self) -> Optional[str]:
        return self._proliferation_key

    @property
    def apoptosis_key(self) -> Optional[str]:
        return self._apoptosis_key

    @proliferation_key.setter
    def proliferation_key(self, value: Optional[str] = None) -> None:
        # TODO(michalk8): add check if present in .obs (if not None)
        self._proliferation_key = value

    @apoptosis_key.setter
    def apoptosis_key(self, value: Optional[str] = None) -> None:
        # TODO(michalk8): add check if present in .obs (if not None)
        self._apoptosis_key = value

    @property
    def cell_costs_source(self) -> pd.DataFrame:
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
        except NotImplementedError:
            raise NotImplementedError("The current solver does not allow this property")

    @property
    def cell_costs_target(self) -> pd.DataFrame:
        # TODO(michalk8): do not raise NotImplementedError
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
        except NotImplementedError:
            raise NotImplementedError("The current solver does not allow this property")

    @property
    def _base_problem_type(self) -> Type[B]:
        return TemporalBaseProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "triu", "tril", "explicit"


class LineageProblem(TemporalProblem):
    def prepare(
        self,
        time_key: str,
        lineage_attr: Mapping[str, Any] = MappingProxyType({}),
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "LineageProblem":
        # TODO(michalk8): use and
        if not len(lineage_attr):
            if "cost_matrices" not in self.adata.obsp:
                raise ValueError(
                    "TODO: default location for quadratic loss is `adata.obsp[`cost_matrices`]` \
                        but adata has no key `cost_matrices` in `obsp`."
                )
        # TODO(michalk8): refactor more
        lineage_attr = dict(lineage_attr)
        lineage_attr.setdefault("attr", "obsp")
        lineage_attr.setdefault("key", "cost_matrices")
        lineage_attr.setdefault("loss", None)
        lineage_attr.setdefault("tag", "cost")
        lineage_attr.setdefault("loss_kwargs", {})
        x = y = lineage_attr

        if joint_attr is None:
            kwargs["callback"] = "local-pca"
        elif isinstance(joint_attr, str):
            kwargs["joint_attr"] = {"x_attr": "obsm", "x_key": joint_attr, "y_attr": "obsm", "y_key": joint_attr}

        return super().prepare(
            time_key,
            x=x,
            y=y,
            policy=policy,
            **kwargs,
        )

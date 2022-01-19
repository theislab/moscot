from typing import Optional, Union, Sequence, Literal, Any, Mapping
from types import MappingProxyType
from anndata import AnnData
import numpy as np
import numpy.typing as npt
import logging
from moscot.solvers._base_solver import BaseSolver
from moscot.utils import _get_marginal, _verify_dict
from moscot.mixins._time_analysis import TemporalAnalysisMixin
from moscot.problems._compound_problem import CompoundProblem


class TemporalProblem(TemporalAnalysisMixin, CompoundProblem):
    # TODO(michalk8): decide how to pass marginals
    # maybe require for BaseProblem as
    # _compute_marginals(self, **kwargs: Any) -> Tuple[Optional[npt.ArrayLike], Optional[npt.ArrayLike]]:
    # as an optional extra step before prepare and in prepare, specify keys

    def __init__(self,
                 adata: AnnData,
                 solver: Optional[BaseSolver] = None):
        self._marginal_container = None
        self._n_growth_rates_estimates = None
        self._a_marg_attr = None
        self._a_marg_key = None
        self._name = None
        super().__init__(adata, solver)

    def prepare(
        self,
        key: str,
        subset: Optional[Sequence[Any]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        x: Optional[Mapping[str, Any]] = None,#TODO: check differnece with mappringproxytype
        y: Optional[Mapping[str, Any]] = None,
        a_marg: Optional[Mapping[str, Any]] = None,
        b_marg: Optional[Mapping[str, Any]] = None,
        n_growth_rates_estimates: Optional[int] = None, # currently, a prior can only be passed to the first transport map
        **kwargs: Any,
    ) -> "BaseProblem":

        if x is None:
            x = {"attr": "X"}
        if y is None:
            y = {"attr": "X"}
        self._n_growth_rates_estimates = n_growth_rates_estimates
        self._a_marg = a_marg
        self._b_marg = b_marg
        if self._n_growth_rates_estimates is not None:
            if policy != "sequential":
                raise ValueError("Growth estimates are only available for sequential transport maps.")
            if self._n_growth_rates_estimates < 1:
                raise ValueError("At least one iteration is required.")
            if self._a_marg is not None:
                _verify_dict(self.adata, self._a_marg)
                if "key" in self._a_marg.keys():
                    self._a_marg_key = self._a_marg["key"]
                    self._marginal_container = getattr(self.adata, self._a_marg["attr"])
                    self._name = self._a_marg["key"]
                else:
                    self._marginal_container = self._adata
                    self._name = self._a_marg["attr"]

                super().prepare(key, subset, policy, x=x, y=y, a_marg=self._a_marg, b_marg=self._b_marg, **kwargs)
            else:
                super().prepare(key, subset, policy, x=x, y=y, **kwargs)
        else:
            super().prepare(key, subset, policy, x=x, y=y, **kwargs)

    def solve(
            self,
            eps: Optional[float] = None,
            alpha: float = 0.5,
            tau_a: Optional[float] = 1.0,
            tau_b: Optional[float] = 1.0,
            **kwargs: Any,
    ) -> "BaseProblem":
        if self._n_growth_rates_estimates is None:
            super().solve(eps, alpha, tau_a, tau_b, **kwargs)
        else:
            if self._a_marg is None:
                self._a_marg = {"attr": "obs", "key": "marginal_a"}
                self._marginal_container = self._adata.obs
                self._name = "marginal_a"
                current_key = f"{self._name}_g1"
                self._marginal_container[current_key] = np.nan
                current_col_index = self._marginal_container.columns.get_loc(current_key)
                for tuple, problem in self._problems.items():
                    mask = self._policy.mask(discard_empty=True)[tuple][0]
                    self._problems[tuple].adata.obs["marginal_a"] = np.ones(sum(mask))/sum(mask)
                    self._marginal_container.iloc[mask, current_col_index] = np.ones(sum(mask))/sum(mask)
                logging.info("No prior distribution was provided. A uniform prior was chosen")
            super().solve(eps, alpha, tau_a, tau_b, a_marg=self._a_marg, **kwargs)
            current_key = f"{self._name}_g1"
            for subset, problem in self._problems.items():
                growth_array = self.solution[subset].solution.transport_matrix.sum(axis=1)
                mask = self._policy.mask(discard_empty=True)[subset][0]
                self._problems[subset].adata.obs[current_key] = growth_array
                self._marginal_container.iloc[mask, current_col_index] = growth_array

            for i in range(1, self._n_growth_rates_estimates):
                a_marg = {"attr": current_key} if len(self._a_marg) == 1 else {"attr": self._a_marg["attr"],
                                                                                 "key": current_key}
                b_marg = {}  # TODO: do we need to adapt this as well?
                current_key = f"{self._name}_g{i+1}"
                self._marginal_container[current_key] = np.nan
                current_col_index = self._marginal_container.columns.get_loc(current_key)
                super().solve(eps, alpha, tau_a, tau_b, a_marg=a_marg, b_marg=b_marg, **kwargs)
                for subset, problem in self._problems.items():
                    growth_array = self.solution[subset].solution.transport_matrix.sum(axis=1)
                    mask = self._policy.mask(discard_empty=True)[subset][0]
                    self._problems[subset].adata.obs[current_key] = growth_array
                    self._marginal_container.iloc[mask, current_col_index] = growth_array



    def estimate_growth_rates(self,
                              n_iterations: int = 3,
                              prior: Optional[Union[str, npt.ArrayLike]] = None,
                              subset: Optional[Sequence[Any]] = None,
                              start: Optional[Any] = None,
                              end: Optional[Any] = None,
                              ):

        data = prior if prior is not None else _get_marginal(self.adata, **self._a_marg)
        for i in range(n_iterations):
            self._apply(data, subset, start, end)




    def estimate_marginals(self):
        pass


from typing import Optional, Union, Sequence, Literal, Any, Mapping
from types import MappingProxyType
from anndata import AnnData
import numpy as np
import numpy.typing as npt
import logging

import moscot.solvers._base_solver
from moscot.solvers._base_solver import BaseSolver
from moscot.utils import _get_marginal, _verify_dict
from moscot.mixins._time_analysis import TemporalAnalysisMixin
from moscot.problems._compound_problem import CompoundProblem
from moscot.problems._base_problem import GeneralProblem


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
        self._temporal_key = None
        self._last_growth_key = None
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
        self._temporal_key = key
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
                self._last_growth_key = current_key

    def push_forward(self,
                     start: Any,
                     end: Any,
                     key_groups: Optional[Literal] = None,
                     groups: Optional[Sequence] = None,
                     mass: Optional[npt.ArrayLike] = None,
                     normalize: bool = True,
                     return_all: bool = False,
                     ) -> Union[Sequence[npt.ArrayLike], npt.ArrayLike]:

        if mass is None:
            pairs = self._policy.chain(start, end)
            start_adata = self._problems[pairs[0]].adata
            mass = self._prepare_transport(start_adata, key_groups, groups)
        return self._apply(mass, start=start, end=end, normalize=normalize, return_all=return_all)

    def push_forward_composed(self,
                              start: Any,
                              end: Any,
                              key_groups: Optional[Literal] = None,
                              groups: Optional[Sequence] = None,
                              mass: Optional[npt.ArrayLike] = None,
                              subset: Optional[Sequence[Any]] = None,
                             normalize: bool = True,
                             return_all: bool = False,
                             ) -> Union[Sequence[npt.ArrayLike], npt.ArrayLike]:
        if mass is None:
            pairs = self._policy.chain(start, end)
            start_adata = self._problems[pairs[0]].adata
            mass = self._prepare_transport(start_adata, key_groups, groups)
        return self._apply(mass, subset, start, end, normalize, return_all, forward=True)

    def pull_back(self,
                  start: Any,
                  end: Any,
                  key_groups: Optional[Literal] = None,
                  groups: Optional[Sequence] = None,
                  mass: Optional[npt.ArrayLike] = None,
                  normalize: bool = True,
                  return_all: bool = False,
                  ) -> Union[Sequence[npt.ArrayLike], npt.ArrayLike]:
        if (start, end) not in self._problems.keys():
            raise ValueError("No transport map computed for {}. Try calling 'push_forward_composed' instead.".format((start, end)))
        if mass is None:
            pairs = self._policy.chain(start, end)
            start_adata = self._problems[pairs[0]].adata
            mass = self._prepare_transport(start_adata, key_groups, groups)
        return self._apply(mass, start=start, end=end, normalize=normalize, return_all=return_all, forward=True)

    def pull_back_composed(self,
                           start: Any,
                           end: Any,
                           key_groups: Optional[Literal] = None,
                           groups: Optional[Sequence] = None,
                           mass: Optional[npt.ArrayLike] = None,
                           subset: Optional[Sequence[Any]] = None,
                           normalize: bool = True,
                           return_all: bool = False,
                           ) -> Union[Sequence[npt.ArrayLike], npt.ArrayLike]:
        if mass is None:
            pairs = self._policy.chain(start, end)
            start_adata = self._problems[pairs[0]].adata
            mass = self._prepare_transport(start_adata, key_groups, groups)
        return self._apply(mass, subset, start, end, normalize, return_all, forward=False)

    def _prepare_transport(self, adata: AnnData, key_groups: Optional[Literal] = None, groups: Optional[Sequence[Any]] = None):
        if key_groups is None:
            mass_length = adata.n_obs
            return np.full(mass_length, 1 / mass_length)
        if key_groups not in self._adata.obs.columns:
            raise ValueError(f"column {key_groups} not found in AnnData.obs")

        self._verify_groups_are_present(groups, key_groups)
        return self._make_mass_from_groups(groups, key_groups)

    def _verify_groups_are_present(self, adata: AnnData, key_groups: Optional[Literal] = None, groups: Optional[Sequence[Any]] = None) -> None:
        adata_groups = adata.obs[key_groups].values
        for group in groups:
            if group not in adata_groups:
                raise ValueError(f"Group {group} is not present for considered data point {start}")

    def _make_mass_from_groups(self, adata: AnnData, key_groups: Optional[Literal] = None, groups: Optional[Sequence[Any]] = None) -> np.ndarray:
        isin_group = adata.obs[key_groups].isin(groups)
        mass = np.zeros(adata.n_obs)
        mass[isin_group] = 1/sum(isin_group)
        return mass

    def get_transport_matrix(self):
        return {subset: self.solution[subset].solution.transport_matrix for subset, problem in self._problems.items()}


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

    def compute_backward_fate(self,
                            adata: AnnData,
                            start: Any,
                            end: Any,
                            key_groups: str,
                            target_cell_populations: Optional[Sequence] = None,):
        """
        given a group of cell types in the target distribution compute their origin
        """
        if (start, end) not in self._problems.keys():
            logging.info("No transport map computed for {}. Trying to compose transports.".format((start, end)))
            pairs = self._policy.chain(start, end)
        else:
            pairs = [(start, end)]
        target_adata = self._problems[pairs[-1]]._adata_y
        if key_groups not in adata.obs.columns:
            raise ValueError("Please provide a key where cell groups are stored")
        if target_cell_populations is None:
            target_cell_populations = list(target_adata.obs[key_groups].values)
        else:
            assert set(target_cell_populations) <= set(target_adata.obs[key_groups].values), "Not all cells given are found at time {}".format(pairs[-1][1])
        mass = self._prepare_transport(target_adata, key_groups, target_cell_populations)
        pulled_back_masses = self._apply(mass, start, end, return_all=True, forward=False)
        #TODO

    def cell_type_transition(self,
                             start: Any,
                             end: Any,
                             start_cell_types: Optional[Sequence] = None,
                             target_cell_types: Optional[Sequence] = None,
                             key_groups: Optional[str] = None,
                             plot=False):
        if (start, end) not in self._problems.keys():
            logging.info("No transport map computed for {}. Trying to compose transports.".format((start, end)))
            pairs = self._policy.chain(start, end)
        else:
            pairs = [(start, end)]
        source_adata = self._problems[pairs[0]].adata
        target_adata = self._problems[pairs[-1]]._adata_y
        if key_groups not in source_adata.obs.columns:
            raise ValueError("Please provide a key where cell groups are stored")
        if key_groups not in target_adata.obs.columns:
            raise ValueError("Please provide a key where cell groups are stored")
        if start_cell_types is None:
            start_cell_types = list(source_adata.obs[key_groups].values)
        else:
            assert set(start_cell_types) <= set(source_adata.obs[key_groups].values), "Not all cells given are found at time {}".format(pairs[-1][1])
        if target_cell_types is None:
            target_cell_types = list(target_adata.obs[key_groups].values)
        else:
            assert set(start_cell_types) <= set(
                source_adata.obs[key_groups].values), "Not all cells given are found at time {}".format(pairs[-1][1])

        mass = self._prepare_transport(source_adata, key_groups, start_cell_types)
        target_mass = self._apply(mass, start, end, normalize=False, return_all=False, forward=True)
        target_mass = target_mass[target_adata.obs[key_groups] in target_cell_types]
        unnormalized_transition = mass @ target_mass
        if plot:
            pass #TODO: make moscot plotting tool for transition matrices
        return unnormalized_transition / unnormalized_transition.sum()

    def validate_by_interpolation(self,
                                  start: Any,
                                  end: Any,
                                  intermediate: Optional[Any] = None,
                                  interpolation_parameter: int = 0.5):
        """
        currently this assumes that we have preprocessed data which results in the questionable assumption that
        the held out data was also used for the preprocessing (whereas it should follow the independent preprocessing
        step of WOT

        Parameters
        ----------
        start
        end
        intermediate
        interpolation_parameter

        Returns
        -------

        """
        # TODO: compute interpolation factor if start and end numeric. Therefore, we first need to allow start and end to be numeric
        if intermediate not in self.adata.obs[self._temporal_key]:
            raise ValueError("No data points corresponding to {} found in adata.obs[{}]".format(intermediate, self._temporal_key))
        if (start, end) not in self._problems.keys():
            raise ValueError("No transport map computed for {}. Trying to compose transports.".format((start, end)))

        source_adata = self._problems[(start, end)].adata
        target_adata = self._problems[(start, end)]._adata_y
        intermediate_adata = self._problems[(start, intermediate)]._adata_y
        gex_ot_interpolated = self._interpolate_gex_with_ot(intermediate_adata.n_obs, source_adata, target_adata, self._problems[(start, end)]._solver)
        distance_gex_ot_interpolated = self._compute_wasserstein_distance(intermediate_adata, AnnData(gex_ot_interpolated))
        gex_randomly_interpolated = self._interpolate_gex_randomly(intermediate_adata.n_obs, source_adata, target_adata)
        distance_gex_randomly_interpolated = self._compute_wasserstein_distance(intermediate_adata, AnnData(gex_randomly_interpolated))
        if self._last_growth_key is not None:
            intermediate_adata.obs[self._last_growth_key]
            gex_randomly_interpolated_growth = self._interpolate_gex_randomly(intermediate_adata.n_obs, source_adata, target_adata, self._problems[(start, end)]._solver, self._marginal_container.iloc[intermediate_indices, -1])
            distance_gex_randomly_interpolated_growth = self._compute_wasserstein_distance((intermediate_adata, AnnData(gex_randomly_interpolated_growth)))
        else:
            distance_gex_randomly_interpolated_growth = None

        return distance_gex_ot_interpolated, distance_gex_randomly_interpolated, distance_gex_randomly_interpolated_growth

    def _interpolate_gex_with_ot(self,
                                 number_cells: int,
                                 adata_1: AnnData,
                                 adata_2: AnnData,
                                 solver: moscot.solvers._base_solver.BaseSolverOutput,
                                 interpolation_parameter: int = 0.5,
                                 adjust_by_growth = True
                                 ):
        #TODO: make online available
        if adjust_by_growth:
            transport_matrix = solver.transport_matrix / np.power(solver.transport_matrix.sum(axis=0), 1. - interpolation_parameter)
        else:
            transport_matrix = solver.transport_matrix
        transport_matrix = transport_matrix.flatten(order='C')
        transport_matrix_flattened = transport_matrix / transport_matrix.sum()
        choices = np.random.choice(adata_1.n_obs * adata_2.n_obs, p=transport_matrix_flattened, size=number_cells)
        return np.asarray([adata_1.X[i // adata_2.n_obs] * (1 - interpolation_parameter) + adata_2.X[i % adata_2.n_obs] * interpolation_parameter for i in choices], dtype=np.float64)

    def _interpolate_gex_randomly(self,
                                  number_cells: int,
                                  adata_1: AnnData,
                                  adata_2: AnnData,
                                  interpolation_parameter: int = 0.5,
                                  growth_rates: Optional[npt.ArrayLike] = None):

        if growth_rates is None:
            choices = np.random.choice(adata_1.n_obs * adata_2.n_obs, size=number_cells)
        else:
            outer_product = np.outer(growth_rates, np.ones(len(growth_rates)))
            outer_product_flattened = outer_product.flatten(order="C")
            outer_product_flattened /= outer_product_flattened.sum()
            choices = np.random.choice(adata_1.n_obs * adata_2.n_obs, p=outer_product_flattened, size=number_cells)

        return np.asarray([adata_1.X[i // adata_2.n_obs] * (1 - interpolation_parameter) + adata_2.X[i % adata_2.n_obs] * interpolation_parameter for i in choices], dtype=np.float64)

    def _compute_wasserstein_distance(self,
                                     adata_1: AnnData = None,
                                     adata_2: AnnData = None,
                                     epsilon: Optional[float] = None,
                                     ):
        problem = GeneralProblem(adata_1, adata_2, solver=self._problems.values()[0]._solver)
        problem.prepare(adata_1, adata_2)
        result = problem.solve(eps=epsilon)
        return result.cost




    def estimate_marginals(self):
        pass


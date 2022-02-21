from typing import Optional, Type, Sequence, Tuple, Any, Union, Literal, Dict
from numbers import Number
from anndata import AnnData
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd


from moscot.problems.time._utils import beta, delta
from moscot.mixins._time_analysis import TemporalAnalysisMixin
from moscot.problems._compound_problem import SingleCompoundProblem
from moscot.problems import MultiMarginalProblem, CompoundProblem
from moscot.solvers._base_solver import BaseSolver
from moscot.solvers._output import BaseSolverOutput

class TemporalProblem(TemporalAnalysisMixin, SingleCompoundProblem):
    
    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None):
        self._temporal_key: str = None
        super().__init__(adata, solver, base_problem_type=TemporalBaseProblem)
        
    def prepare(self, 
                key: str, 
                data_key: str = "X_pca",
                policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential", 
                subset: Optional[Sequence[Tuple[Any, Any]]] = None, 
                a: Optional[Union[npt.ArrayLike, str]] = None,
                b: Optional[Union[npt.ArrayLike, str]] = None,
                estimate_marginals: bool = False,
                gene_set_proliferation: Sequence = None,
                gene_set_apoptosis: Sequence = None,
                obs_proliferation_str: str = "obs_proliferation",
                obs_apoptosis_str: str = "obs_apoptosis",
                reference: Optional[Any] = None, 
                marginal_kwargs: Dict[Any, Any] = {},
                **kwargs: Any) -> "CompoundProblem":
        
        self._temporal_key = key
        if estimate_marginals:
            self._score_gene_sets(gene_set_proliferation, obs_proliferation_str, **kwargs)
            self._score_gene_sets(gene_set_apoptosis, obs_apoptosis_str, **kwargs)
            if a is not None or b is not None:
                raise Warning("The marginals are estimated by proliferation and apoptosis genes.")
            a = True
            b = True
            marginal_kwargs["obs_proliferation_str"] = obs_proliferation_str
            marginal_kwargs["obs_apoptosis_str"] = obs_apoptosis_str
        
        x = {"attr": "obsm", "key": data_key}
        y = {"attr": "obsm", "key": data_key}

        return super().prepare(key, policy, subset, reference, x=x, y=y, a=a, b=b, add_metadata=True, marginal_kwargs=marginal_kwargs, **kwargs)

    def solve(self,
              epsilon: float = 0.5,
              tau_a: float = 1.0,
              tau_b: float = 1.0,
              n_iters: int = 1,
              online: bool = False,
              solve_kwargs: Dict[Any, Any] = {},
              **kwargs: Any):
        
        super().solve(epsilon=epsilon, tau_a=tau_a, tau_b=tau_b, online=online, n_iters=n_iters, solve_kwargs=solve_kwargs, **kwargs)

    def push_forward(
        self,
        start: Any,
        end: Any,
        mass: Optional[npt.ArrayLike] = None,
        key_groups: Optional[str] = None,
        groups: Optional[Sequence] = None,
        normalize: bool = True,
        return_all: bool = False,
        result_col: str = "push_result",
        return_result: bool = False
    ) -> Optional[Union[Sequence[npt.ArrayLike], npt.ArrayLike]]:

        if (start, end) not in self._problems.keys():
            raise ValueError(
                f"No transport map computed for {(start, end)}. Try calling 'push_forward_composed' instead."
            )
        if mass is None:
            result = self._apply(
                data=key_groups,
                subset=groups,
                start=start,
                end=end,
                normalize=normalize,
                return_all=return_all,
                forward=True,
                scale_by_marginals=True,
                return_as_dict=True,
            )[start, end]
        else:
            result = self._apply(
                data=mass, start=start, end=end, normalize=normalize, return_all=return_all, forward=True, scale_by_marginals=True, return_as_dict=True
            )[start, end]
        self._dict_to_adata(result, result_col)
        if return_result:
            return result

    def push_forward_composed(
        self,
        start: Any,
        end: Any,
        mass: Optional[npt.ArrayLike] = None,
        key_groups: Optional[str] = None,
        groups: Optional[Sequence] = None,
        normalize: bool = True,
        return_all: bool = False,
        result_col: str = "push_result",
        return_result: bool = False
    ) -> Union[Sequence[npt.ArrayLike], npt.ArrayLike]:
        if mass is None:
            result = self._apply(
                data=key_groups,
                subset=groups,
                start=start,
                end=end,
                normalize=normalize,
                return_all=True,
                forward=True,
                return_as_dict=True,
            )[start, end]
        else:
            result = self._apply(
                data=mass,
                start=start,
                end=end,
                normalize=normalize,
                return_all=return_all,
                forward=True,
                return_as_dict=True,
            )[start, end]
        self._dict_to_adata(result, result_col)
        if return_result:
            return result
        
    def pull_back(
        self,
        start: Any,
        end: Any,
        mass: Optional[npt.ArrayLike] = None,
        key_groups: Optional[str] = None,
        groups: Optional[Sequence] = None,
        normalize: bool = True,
        return_all: bool = False,
        result_col: str = "pull_result",
        return_result: bool = False
    ) -> Optional[Union[Sequence[npt.ArrayLike], npt.ArrayLike]]:

        if (start, end) not in self._problems.keys():
            raise ValueError(
                f"No transport map computed for {(start, end)}. Try calling 'pull_back_composed' instead."
            )
        if mass is None:
            result = self._apply(
                data=key_groups,
                subset=groups,
                start=start,
                end=end,
                normalize=normalize,
                return_all=return_all,
                forward=False,
                scale_by_marginals=True,
                return_as_dict=True,
            )[start, end]
        else:
            result = self._apply(
                data=mass, start=start, end=end, normalize=normalize, return_all=return_all, forward=False, scale_by_marginals=True, return_as_dict=True
            )[start, end]
        self._dict_to_adata(result, result_col)
        if return_result:
            return result

    def pull_back_composed(
        self,
        start: Any,
        end: Any,
        mass: Optional[npt.ArrayLike] = None,
        key_groups: Optional[str] = None,
        groups: Optional[Sequence] = None,
        normalize: bool = True,
        return_all: bool = False,
        result_col: str = "pull_result",
        return_result: bool = False
    ) -> Union[Sequence[npt.ArrayLike], npt.ArrayLike]:
        if mass is None:
            result = self._apply(
                data=key_groups,
                subset=groups,
                start=start,
                end=end,
                normalize=normalize,
                return_all=True,
                forward=False,
                scale_by_marginals=True,
                return_as_dict=True,
            )[start, end]
        else:
            result = self._apply(
                data=mass,
                start=start,
                end=end,
                normalize=normalize,
                return_all=return_all,
                forward=False,
                scale_by_marginals=True,
                return_as_dict=True,
            )[start, end]
        self._dict_to_adata(result, result_col)
        if return_result:
            return result

    def _dict_to_adata(self,
                       d: Dict,
                       obs_col: str):
        tmp = np.zeros(len(self.adata))
        for key, value in d.items():
            mask = self.adata.obs[self._temporal_key] == key
            tmp[mask] = np.squeeze(value)
        self.adata.obs[obs_col] = tmp

    def _score_gene_sets(self, obs_str: str, gene_set: Optional[Sequence], max_z_score: int = 5, **kwargs: Any):
        if gene_set is None:
            self.adata.obs[obs_str] = 0
        else:
            self._score_gene_set(obs_str, gene_set, max_z_score)

    def _score_gene_set(self, obs_str: str, gene_set: Sequence, max_z_score: int = 5): #TODO(@MUCDK): Check whether we want to use sc.tl.score_genes() + add other options as in WOT
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
        key_groups: Any,
        early_cell_types: Optional[Sequence] = None,
        late_cell_types: Optional[Sequence] = None,
        plot=True,
        forward=False, # return value will be row-stochastic if forward=True, else column-stochastic
    ) -> npt.ArrayLike:
        if late_cell_types is None:
            late_cell_types = list(self.adata.obs[key_groups].unique())
        if early_cell_types is None:
            early_cell_types = list(self.adata.obs[key_groups].unique())

        transition_table = pd.DataFrame(
            np.zeros((len(early_cell_types), len(late_cell_types))), index=early_cell_types, columns=late_cell_types
        )
        df_early = self.adata[self.adata.obs[self._temporal_key] == start].obs.copy()
        df_late = self.adata[self.adata.obs[self._temporal_key] == end].obs.copy()

        subsets = early_cell_types if forward else late_cell_types
        for i, subset in enumerate(subsets):
            try:
                result = self._apply(
                    data=key_groups,
                    subset=subset,
                    start=start,
                    end=end,
                    normalize=True,
                    return_all=False,
                    forward=forward,
                    scale_by_marginals=True,
                    return_as_dict=False,
                )[start, end]
            except ValueError:
                result = 0
            if forward:
                df_late.loc[:, "distribution"] = result
                target_cell_dist = (
                    df_late[df_late[key_groups].isin(late_cell_types)][[key_groups, "distribution"]]
                    .groupby(key_groups)
                    .sum()
                )
                transition_table.loc[subset, :] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else 0
                    for cell_type in late_cell_types
                ]
            else:
                df_early.loc[:, "distribution"] = result
                target_cell_dist = (
                    df_early[df_early[key_groups].isin(early_cell_types)][[key_groups, "distribution"]]
                    .groupby(key_groups)
                    .sum()
                )
                transition_table.loc[:, subset] = [
                    target_cell_dist.loc[cell_type, "distribution"]
                    if cell_type in target_cell_dist.distribution.index
                    else 0
                    for cell_type in early_cell_types
                ]
        transition_table = transition_table.to_numpy()

        if plot:
            _, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(transition_table)

            ax.set_xticks(np.arange(transition_table.shape[1]))
            ax.set_yticks(np.arange(transition_table.shape[0]))
            ax.set_xticklabels(late_cell_types)
            ax.set_yticklabels(early_cell_types)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(transition_table.shape[0]):
                for j in range(transition_table.shape[1]):
                    _ = ax.text(j, i, f"{transition_table[i, j]:.5f}", ha="center", va="center", color="w")
        return transition_table

    @property
    def transport_matrix(self):
        return {pair: solution._solution.transport_matrix for pair, solution in self._solutions.items()}

    @property
    def solution(self):
        return {pair: solution._solution for pair, solution in self._solutions.items()}


class TemporalBaseProblem(MultiMarginalProblem):
    def __init__(self, adata_x: AnnData, adata_y: AnnData, solver: Type[BaseSolver], metadata: Tuple[Number, Number]):
        self._t_start = metadata[0]
        self._t_end = metadata[1]
        super().__init__(adata_x, adata_y=adata_y, solver=solver)

    def _estimate_marginals(self, adata: AnnData, a: bool, obs_proliferation_str: str, obs_apoptosis_str: str, **kwargs: Any) -> npt.ArrayLike:
        if a:
            birth = beta(adata.obs[obs_proliferation_str], **kwargs)
            death = delta(adata.obs[obs_apoptosis_str], **kwargs)
            return np.exp((birth-death)*(self._t_end-self._t_start))
        else:
            return np.average(self._get_last_marginals[0])

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        self._a.append(np.asarray(sol.transport_matrix.sum(axis=1)))
        self._b.append(np.average(np.asarray(self._get_last_marginals()[0])))

    @property
    def growth_rates(self) -> npt.ArrayLike:
        return [np.power(self._a[i], 1/(self._t_end - self._t_start)) for i in range(len(self._a))]
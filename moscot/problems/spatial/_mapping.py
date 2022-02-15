from types import MappingProxyType
from typing import Any, Dict, Type, Tuple, Union, Iterator, Optional, Sequence, Mapping
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    
from moscot.problems._compound_problem import CompoundProblem
from moscot.problems._base_problem import GeneralProblem
from anndata import AnnData
from typing import Optional
from moscot.solvers._base_solver import BaseSolver

import numpy.typing as npt


class SpatialMappingProblem(CompoundProblem):

    def __init__(self, adata_sc: AnnData, adata_sp: Optional[AnnData] = None,
                 solver: Optional[BaseSolver] = None):
        self._adata_sc = adata_sc
        self._adata_sp = adata_sp
        self._subsets = None
        if adata_sp is not None:
            self._spatial_ref = (adata_sp is not None)
            super().__init__(adata_sc, adata_sp, solver=solver)
        else:
            super().__init__(adata_sc, solver=solver)

    def prepare(
            self,
            sc_attr: Mapping[str, Any],  # ={'atrr':'obsm', 'key':'X_pca'},
            sp_attr: Optional[Mapping[str, Any]] = None,  # = {'atrr': 'obsm', 'key': 'spatial'},
            atlas_attr: Optional[Mapping[str, Any]] = None,
            keys_subset: Optional[Union[str, Dict[Any, Tuple[str, str]]]] = None,
            var_subset: Optional[Union[Tuple[Any, Any], Dict[Any, Tuple[Any, Any]]]] = None,
            a_marg: Optional[Mapping[str, Any]] = MappingProxyType({}),
            b_marg: Optional[Mapping[str, Any]] = MappingProxyType({}),
            **kwargs: Any,
    ) -> "BaseProblem":
        """
        prepare the spatial mapping problem
        Parameters
        ----------
        keys_subset: key(s) for .var which indicate marker genes (expects identical keys for `spatial` and 'scRNA' adata)
        var_subset: subset(s) of marker genes to use, either a single list or dictionary of lists
        sc_attr: dict like for values to take to compute single-cell cost, e.g. {'atrr':'obsm', 'key':'X_pca'}
        sp_attr: dict like for values to take to compute spatial cost, e.g. {'atrr':'obsm', 'key':'spatial'}
        atlas_attr: dict like for values to take to compute atlas joint cost,
                    e.g. {'sc_atrr':'X', 'sp_attr':'X'}
        Returns
        -------

        """
        # TODO(ZP): (1) add `policy` like option; random sampling of genes, increasing number of markers etc.
        #           (2) allow different `keys` for sc and sp
        #           (3) allow single key in `varm`
        self._subsets = var_subset
        if self._spatial_ref:
            if keys_subset is not None:
                if not isinstance(keys_subset, Mapping):
                    self._subsets = {key_: (self._adata_sc.var[key_], self._adata_sp.var[key_])
                                 for key_ in keys_subset}
                else:
                    self._subsets = {subset: (self._adata_sc.var[key_], self._adata_sp.var[key_])
                                 for subset, key_ in keys_subset.items()}
                    
            self._problems = {subset: GeneralProblem(self._adata_sc[:, sc_mask],
                                                     self._adata_sp[:, sp_mask],
                                                     solver=self._solver).prepare(x=sc_attr,
                                                                                  y=sp_attr,
                                                                                  xy=atlas_attr,
                                                                                  a_marg=sc_attr,
                                                                                  b_marg=sp_attr,
                                                                                  **kwargs)
                              for subset, (sc_mask, sp_mask) in self._subsets.items()}
        else:
            self._problems = {'denovo': GeneralProblem(self.adata,
                                                       self.adata,
                                                       solver=self._solver).prepare(x=sc_attr,
                                                                                y=sp_attr,
                                                                                a_marg=sc_attr,
                                                                                b_marg=sp_attr,
                                                                                **kwargs)}

        return self

    # TODO(ZP) validate that solvers use alpha properly and pass epsilons.
    #  + incorporate Low Rank
    def solve(
            self,
            eps: Optional[float] = None,
            alpha: float = 0.5,
            tau_a: Optional[float] = 1.0,
            tau_b: Optional[float] = 1.0,
            **kwargs: Any,
    ) -> "BaseProblem":

        return super().solve(eps=eps, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs)

    def _apply(self, data: Optional[Union[str, npt.ArrayLike]] = None, subset: Optional[Sequence[Any]] = None,
               problems_keys: Optional[Sequence[Any]] = None, normalize: bool = True, forward: bool = True, **kwargs) -> npt.ArrayLike:
        if problems_keys is None:
            problems_keys = self._problems.keys()

        res = []
        for problem_key in problems_keys:
            problem = self._problems[problem_key]
            adata = problem.adata if forward or problem._adata_y is None else problem._adata_y
            data_pk = [problem._get_mass(adata, data, subset=subset, normalize=True)]
            res.append(
                (problem.push if forward else problem.pull)(data_pk, subset=subset, normalize=normalize))

        return res
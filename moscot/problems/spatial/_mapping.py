from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

try:
    pass
except ImportError:
    pass

from typing import Optional

from scanpy import logging as logg

import numpy.typing as npt

from anndata import AnnData

from moscot.backends.ott import GWSolver, FGWSolver
from moscot.problems._base_problem import BaseProblem, GeneralProblem
from moscot.problems._compound_problem import CompoundProblem


class SpatialMappingProblem(CompoundProblem):
    def __init__(
        self,
        adata_sc: AnnData,
        adata_sp: AnnData = None,
        use_reference: bool = False,
        var_names: List[str] | bool | None = None,
        rank: Optional[int] = None,
        solver_jit: Optional[bool] = None,
    ):

        adata_sc, adata_sp = self.filter_vars(adata_sc, adata_sp, var_names)
        self._adata_sc = adata_sc
        self._adata_sp = adata_sp
        self.use_reference = use_reference

        solver = FGWSolver(rank=rank, jit=solver_jit) if use_reference else GWSolver(rank=rank, jit=solver_jit)
        super().__init__(adata_sc, adata_sp, solver=solver)
        self._base_problem_type = GeneralProblem(
            self._adata_sc,
            self._adata_sp,
            solver=solver,
        )

    @property
    def adata_sp(self):
        return self._adata_sp

    @property
    def adata_sc(self):
        return self._adata_sc

    @property
    def problems(self):
        return self._problems

    def filter_vars(
        self,
        adata_sc: AnnData,
        adata_sp: AnnData,
        var_names: Optional[List[str]] = None,
    ) -> Tuple[AnnData, AnnData]:
        vars_sc = set(adata_sc.var_names)  # TODO: allow alternative gene symbol by passing var_key
        vars_sp = set(adata_sp.var_names)
        var_names = set(var_names) if var_names is not None else None
        if var_names is None:
            var_names = vars_sp.intersection(vars_sc)
            if len(var_names):
                return adata_sc[:, list(var_names)], adata_sp[:, list(var_names)]
            else:
                logg.warning(f"`adata_sc` and `adata_sp` do not share `var_names`. ")
                return adata_sc, adata_sp
        else:
            if var_names.issubset(vars_sc) and var_names.issubset(vars_sp):
                return adata_sc[:, list(var_names)], adata_sp[:, list(var_names)]
            else:
                raise ValueError("Some `var_names` ares missing in either `adata_sc` or `adata_sp`.")

    def prepare(
        self,
        attr_sc: Mapping[str, Any] = {"attr": "obsm", "key": "X_scvi"},
        attr_sp: Optional[Mapping[str, Any]] = {"attr": "obsm", "key": "spatial"},
        attr_joint: Optional[Mapping[str, Any]] = {"x_attr": "X", "y_attr": "X"},
        **kwargs: Any,
    ) -> BaseProblem:
        # TODO(ZP): (1) add `policy` like option; random sampling of genes, increasing number of markers etc.

        if self.use_reference:
            super().prepare(x=attr_sc, y=attr_sp, xy=attr_joint, **kwargs)
        else:
            super().prepare(x=attr_sc, y=attr_sp, **kwargs)

    def solve(
        self,
        eps: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        **kwargs: Any,
    ) -> BaseProblem:

        super().solve(eps=eps, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs)

    def _apply(  # TODO: do we need this apply ?
        self,
        data: Optional[Union[str, npt.ArrayLike]] = None,
        subset: Optional[Sequence[Any]] = None,
        problems_keys: Optional[Sequence[Any]] = None,
        normalize: bool = True,
        forward: bool = True,
        **kwargs,
    ) -> npt.ArrayLike:
        if problems_keys is None:
            problems_keys = self._problems.keys()

        res = {}
        for problem_key in problems_keys:
            problem = self._problems[problem_key]
            adata = problem.adata if forward or problem._adata_y is None else problem._adata_y
            data_pk = [problem._get_mass(adata, data, subset=subset, normalize=True)]
            res[problem_key] = (problem.push if forward else problem.pull)(data_pk, subset=subset, normalize=normalize)

        return res

from __future__ import annotations

from typing import Any, List, Tuple, Mapping, Optional

try:
    pass
except ImportError:
    pass

from typing import Optional

from scanpy import logging as logg

from anndata import AnnData

from moscot.backends.ott import GWSolver, FGWSolver
from moscot.problems._base_problem import GeneralProblem
from moscot.problems._compound_problem import SingleCompoundProblem


class SpatialMappingProblem(GeneralProblem):
    def __init__(
        self,
        adata_sc: AnnData,
        adata_sp: AnnData = None,
        use_reference: bool = False,
        var_names: List[str] | bool | None = None,
        rank: Optional[int] = None,
        solver_jit: Optional[bool] = None,
    ):
        # save full adatas for later analysis
        self._adata_sc = adata_sc
        self._adata_sp = adata_sp

        adata_sc_ref, adata_sp_ref = self.filter_vars(adata_sc, adata_sp, var_names, use_reference)
        solver = FGWSolver(rank=rank, jit=solver_jit) if use_reference else GWSolver(rank=rank, jit=solver_jit)
        super().__init__(adata_sc_ref, adata_sp_ref, solver=solver)
        self.use_reference = use_reference

    @property
    def adata_sp(
        self,
    ) -> AnnData:
        return self._adata_sp

    @property
    def adata_sc(self) -> AnnData:
        return self._adata_sc

    @property
    def problems(self) -> GeneralProblem:
        return self._problems

    def filter_vars(
        self,
        adata_sc: AnnData,
        adata_sp: AnnData,
        var_names: Optional[List[str]] = None,
        use_reference: bool = False,
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
            if not use_reference:
                return adata_sc, adata_sp
            elif use_reference and var_names.issubset(vars_sc) and var_names.issubset(vars_sp):
                return adata_sc[:, list(var_names)], adata_sp[:, list(var_names)]
            else:
                raise ValueError("Some `var_names` ares missing in either `adata_sc` or `adata_sp`.")

    def prepare(
        self,
        attr_sc: Mapping[str, Any] = {"attr": "obsm", "key": "X_scvi"},
        attr_sp: Optional[Mapping[str, Any]] = {"attr": "obsm", "key": "spatial"},
        attr_joint: Optional[Mapping[str, Any]] = {"x_attr": "X", "y_attr": "X"},
        **kwargs: Any,
    ) -> GeneralProblem:

        if self.use_reference:
            return super().prepare(x=attr_sc, y=attr_sp, xy=attr_joint, **kwargs)
        else:
            return super().prepare(x=attr_sc, y=attr_sp, **kwargs)

    def solve(
        self,
        eps: Optional[float] = None,
        alpha: float = 0.5,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        **kwargs: Any,
    ) -> GeneralProblem:

        return super().solve(eps=eps, alpha=alpha, tau_a=tau_a, tau_b=tau_b, **kwargs)

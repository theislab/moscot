from types import MappingProxyType
from typing import Any, Dict, Type, Tuple, Union, Iterator, Optional, Sequence, Mapping
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    
from moscot.problems._compound_problem import CompoundProblem
from moscot.problems._base_problem import BaseProblem, GeneralProblem

from moscot.backends.ott import GWSolver, FGWSolver

from anndata import AnnData
from typing import Optional

from moscot.solvers._output import BaseSolverOutput
from moscot.problems._anndata import AnnDataPointer
from moscot.solvers._base_solver import BaseSolver
from moscot.solvers._tagged_array import Tag, TaggedArray

import numpy.typing as npt


class SpatialGeneralProblem(GeneralProblem):
    def __init__(
        self,
        adata_sc: AnnData,
        adata_sp: Optional[AnnData] = None,
        adata_atlas: Optional[AnnData] = None,
        solver: Optional[BaseSolver] = None,
    ):
        super().__init__(adata_sc, adata_sp, adata_atlas, solver)

    def prepare(
        self,
        sc: Mapping[str, Any] = MappingProxyType({}),
        sp: Optional[Mapping[str, Any]] = None,
        atlas: Optional[Mapping[str, Any]] = None,
        a: Optional[Union[str, npt.ArrayLike]] = None,
        b: Optional[Union[str, npt.ArrayLike]] = None,
        **kwargs: Any,
    ) -> "GeneralProblem":
        self._x = AnnDataPointer(adata=self.adata, **sc).create()
        self._y = None if sp is None else AnnDataPointer(adata=self._adata_y, **sp).create()
        self._xy = None if atlas is None else self._handle_joint(**atlas)

        self._a = BaseProblem._get_or_create_marginal(self.adata, a)
        self._b = BaseProblem._get_or_create_marginal(self._marginal_b_adata, b)
        self._solution = None

        return self

    def _handle_joint(
        self, create_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any
    ) -> Optional[Union[TaggedArray, Tuple[TaggedArray, TaggedArray]]]:
        if not isinstance(self._solver, FGWSolver):
            return None

        sc_kwargs = {k[3:]: v for k, v in kwargs.items() if k.startswith("sc_")}
        sp_kwargs = {k[3:]: v for k, v in kwargs.items() if k.startswith("sp_")}

        sc_mask = sc_kwargs.pop("mask", self.adata.var_names)
        sp_mask = sp_kwargs.pop("mask", self._adata_y.var_names)
        tag = kwargs.get("tag", None)
        if tag is None:
            # TODO(michalk8): better/more strict condition?
            # TODO(michalk8): specify which tag is being using
            tag = Tag.POINT_CLOUD if "sc_attr" in kwargs and "sp_attr" in kwargs else Tag.COST_MATRIX

        tag = Tag(tag)
        if tag in (Tag.COST_MATRIX, Tag.KERNEL):
            attr = kwargs.get("attr", "X")
            if attr == "obsm":
                return AnnDataPointer(self.adata, tag=tag, **kwargs).create(**create_kwargs)
            if attr == "varm":
                kwargs["attr"] = "obsm"
                return AnnDataPointer(self._adata_y.T, tag=tag, **kwargs).create(**create_kwargs)
            if attr not in ("X", "layers", "raw"):
                raise AttributeError("TODO: expected obsm/varm/X/layers/raw")
            if self._adata_xy is None:
                raise ValueError("TODO: Specifying cost/kernel requires joint adata.")
            return AnnDataPointer(self._adata_xy, tag=tag, **kwargs).create(**create_kwargs)
        if tag != Tag.POINT_CLOUD:
            # TODO(michalk8): log-warn
            tag = Tag.POINT_CLOUD


        sc_array = AnnDataPointer(self.adata[:, sc_mask], tag=tag, **sc_kwargs).create(**create_kwargs)
        sp_array = AnnDataPointer(self._adata_y[:, sp_mask], tag=tag, **sp_kwargs).create(**create_kwargs)

        return sc_array, sp_array

class SpatialMappingProblem(CompoundProblem):

    def __init__(self, adata_sc: AnnData,
                 adata_sp: Optional[AnnData] = None,
                 spatial_ref: Optional[Union[bool, AnnData]] = None,
                 solver: Optional[BaseSolver] = None):
        self._adata_sc = adata_sc
        self._adata_sp = adata_sp
        self._spatial_ref = (spatial_ref is not None)
        self._adata_atlas = None
        if isinstance(spatial_ref, AnnData):
            self._adata_atlas = spatial_ref
        self._subsets = None
        solver = solver if (solver is not None) else FGWSolver() if self._spatial_ref else GWSolver()

        if adata_sp is not None:
            super().__init__(adata_sc, adata_sp, solver=solver)
        else:
            super().__init__(adata_sc, solver=solver)

    # TODO (ZP) return copies ?
    @property
    def adata_sp(self):
        return self._adata_sp

    @property
    def adata_sc(self):
        return self._adata_sc

    @property
    def problems(self):
        return self._problems

    def prepare(
            self,
            sc_attr: Mapping[str, Any],  # ={'atrr':'obsm', 'key':'X_pca'},
            sp_attr: Optional[Mapping[str, Any]] = None,  # = {'atrr': 'obsm', 'key': 'spatial'},
            atlas_attr: Optional[Mapping[str, Any]] = None,
            keys_subset: Optional[Union[str, Dict[Any, Tuple[str, str]]]] = None,
            var_subset: Optional[Union[Tuple[Any, Any], Dict[Any, Tuple[Any, Any]]]] = None,
            a_marg: Optional[Mapping[str, Any]] = None,
            b_marg: Optional[Mapping[str, Any]] = None,
            **kwargs: Any,
    ) -> "BaseProblem":
        """
        prepare the spatial mapping problem
        Parameters
        ----------
        keys_subset: key(s) for .var which indicate marker genes (expects identical keys for `spatial` and 'scRNA' adata)
        var_subset: subset(s) of marker genes to use, either a single list or dictionary of lists
        sc_attr: dict like for values to take to compute single-cell cost, e.g. {'attr':'obsm', 'key':'X_pca'}
        sp_attr: dict like for values to take to compute spatial cost, e.g. {'attr':'obsm', 'key':'spatial'}
        atlas_attr: dict like for values to take to compute atlas joint cost,
                    e.g. {'sc_atrr':'X', 'sp_attr':'X'}
        a_marg: Optional[Mapping[str, Any]] marginals ,
        b_marg: Optional[Mapping[str, Any]] marginals,
        Returns
        -------

        """
        # TODO(ZP): (1) add `policy` like option; random sampling of genes, increasing number of markers etc.
        #           (2) allow different `keys` for sc and sp
        #           (3) allow single key in `varm`

        if self._spatial_ref:
            self._subsets = var_subset
            if keys_subset is not None:
                if not isinstance(keys_subset, Mapping):
                    self._subsets = {key_: (self._adata_sc.var[key_], self._adata_sp.var[key_])
                                     for key_ in keys_subset}
                else:
                    self._subsets = {subset: (self._adata_sc.var[key_], self._adata_sp.var[key_])
                                     for subset, key_ in keys_subset.items()}

        if self._subsets is not None:
            sc_attr_ref = atlas_attr['sc_attr'] if 'sc_attr' in atlas_attr else 'X'
            sp_attr_ref = atlas_attr['sp_attr'] if 'sp_attr' in atlas_attr else 'X'
            self._problems = {subset: SpatialGeneralProblem(self._adata_sc,
                                                            self._adata_sp,
                                                            self._adata_atlas,
                                                            solver=self._solver).prepare(sc=sc_attr,
                                                                                         sp=sp_attr,
                                                                                         atlas={'sc_mask': sc_mask,
                                                                                                'sp_mask': sp_mask,
                                                                                                'sc_attr': sc_attr_ref,
                                                                                                'sp_attr': sp_attr_ref},
                                                                                         a_marg=a_marg,
                                                                                         b_marg=b_marg,
                                                                                         **kwargs)
                              for subset, (sc_mask, sp_mask) in self._subsets.items()}

        else:
            self._problems = {'denovo': SpatialGeneralProblem(self._adata_sc,
                                                              self._adata_sp,
                                                              solver=self._solver).prepare(x=sc_attr,
                                                                                           y=sp_attr,
                                                                                           a_marg=a_marg,
                                                                                           b_marg=b_marg,
                                                                                           **kwargs)}

        print(f'done preparing Spatial Problem with {len(self._problems)} problems')

        return self

    # TODO(ZP) validate that solver s use alpha properly and pass epsilons.
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

        res = {}
        for problem_key in problems_keys:
            problem = self._problems[problem_key]
            adata = problem.adata if forward or problem._adata_y is None else problem._adata_y
            data_pk = [problem._get_mass(adata, data, subset=subset, normalize=True)]
            res[problem_key] = (problem.push if forward else problem.pull)(data_pk, subset=subset, normalize=normalize)

        return res


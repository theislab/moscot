from types import MappingProxyType
from typing import Any, Dict, Tuple, Union, Literal, Mapping, Callable, Optional, Sequence
from numbers import Number
import logging

import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData
import scanpy as sc

from moscot._docs import d
from moscot.problems import MultiMarginalProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._utils import beta, delta, MarkerGenes
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
        return np.full(len(self._marginal_b_adata), np.average(growth))

    def _add_marginals(self, sol: BaseSolverOutput) -> None:
        _a = np.asarray(sol.a) / self._a[-1]
        self._a.append(_a)
        self._b.append(np.full(len(self._marginal_b_adata), np.average(_a)))

    @property
    def growth_rates(self) -> npt.ArrayLike:
        return np.power(self.a, 1 / (self._target - self._source))


class TemporalProblem(TemporalAnalysisMixin, SingleCompoundProblem):
    """
    Estimator for modelling time series single cell data based on :cite:`schiebinger:19`

    Class handling the computation and downstream analysis of temporal single cell data.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` instance containing the single cell data and corresponding metadata
    solver
        :class:`moscot.solver` instance used to solve the optimal transport problem. Currently, :class:`moscot.solvers.SinkhornSolver` can be used to solve this problem.

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it
    """

    _VALID_POLICIES = ["sequential", "pairwise", "triu", "tril", "explicit"]

    def __init__(self, adata: AnnData, solver: Optional[BaseSolver] = None, **kwargs: Any):
        super().__init__(adata, solver=solver, base_problem_type=TemporalBaseProblem, **kwargs)
        self._temporal_key: Optional[str] = None
        self._proliferation_key: Optional[str] = None
        self._apoptosis_key: Optional[str] = None

    @d.dedent
    def score_genes_for_marginals(
        self,
        gene_set_proliferation: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        gene_set_apoptosis: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        proliferation_key: str = "proliferation",
        apoptosis_key: str = "apoptosis",
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Compute gene scores to obtain prior konwledge about proliferation and apoptosis.

        This method computes gene scores using :func:`scanpy.tl.score_genes`. Therefore, a list of genes corresponding to proliferation and/or apoptosis must be passed.
        ALternatively, proliferation and apoptosis genes for humans and mice are saved in :mod:`moscot`.
        The gene scores will be used in :meth:`moscot.problems.TemporalProblem.prepare()` to estimate the initial growth rates as suggested in :cite:`schiebinger:19`

        Parameters
        ----------
        gene_set_proliferation
            Set of marker genes for proliferation used in the birth-death process. If marker genes from :mod:`moscot` are to be used the corresponding organism must be passed.
        gene_set_apoptosis
            Set of marker genes for apoptosis used in the birth-death process. If marker genes from :mod:`moscot` are to be used the corresponding organism must be passed.
        proliferation_key
            Key in :attr:`anndata.AnnData.obs` where to add the genes scores.
        kwargs
            Keyword arguments for :func:`scanpy.tl.score_genes`.

        Returns
        -------
        Self and updates the following attributes

            - :attr:`proliferation_key`
            - :attr:`apoptosis_key`

        Notes
        -----
        The marker genes in :mod:`moscot` are taken from the following sources:
            - human, proliferation - :cite:`tirosh:16:science`.
            - human, apoptosis - `Hallmark Apoptosis, MSigDB <https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_APOPTOSIS>`_.
            - mouse, proliferation - :cite:`tirosh:16:nature`.
            - mouse, apoptosis - `Hallmark P53 Pathway, MSigDB <https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_P53_PATHWAY>`_.

        """
        if gene_set_proliferation is None:
            self.proliferation_key = None
        else:
            if isinstance(gene_set_proliferation, str):
                sc.tl.score_genes(
                    self.adata,
                    getattr(MarkerGenes, "proliferation_markers")(gene_set_proliferation),
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
                    getattr(MarkerGenes, "apoptosis_markers")(gene_set_apoptosis),
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

    @d.dedent
    def prepare(
        self,
        time_key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "TemporalProblem":
        """
        Prepares the TemporalProblem for it being ready to be solved

        This method executes multiple steps to prepare the problem for the Optimal Transport solver to be ready to solve it

        Parameters
        ----------
        %(CompoundBaseProblem_prepare.parameters)s
        time_key
            Key in :attr:`anndata.AnnData.obs` which defines the time point each cell belongs to. It is supposed to be of numerical data type.
        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps. If None, the data is read from :attr:`anndata.AnnData.X` and 
            for each time point the corresponding PCA space is computed. If `joint_attr` is a string the data is assumed to be found in :attr:`anndata.AnnData.obsm`. 
            If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of :attr:`anndata.AnnData` as a key and the corresponding attribute as a value.

        Returns
        -------
        :class:`moscot.problems.time.LineageProblem`

        Raises
        ------
        KeyError
            If `time_key` is not in :attr:`anndata.AnnData.obs`
        KeyError
            If `joint_attr` is a string and cannot be found in :attr:`anndata.AnnData.obsm`
        """
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

    @d.dedent
    def push(
        self,
        start: Number,
        end: Number,
        result_key: Optional[str] = None,
        return_all: bool = False,
        scale_by_marginals: bool = True,
        **kwargs: Any,
    ) -> Optional[Union[npt.ArrayLike, Dict[Tuple[Any, Any], npt.ArrayLike]]]:
        """
        Pushes distribution of cells through time

        Parameters
        ----------
        start
            Time point of source distribution.
        target
            Time point of target distribution.
        result_key
            Key of where to save the result in :attr:`anndata.AnnData.obs`. If None the result will be returned.
        return_all
            If True returns all the intermediate masses if pushed through multiple transport plans. If True, the result is returned as a dictionary.

        Returns
        -------
        Depending on `result_key` updates `adata` or returns the result. In the latter case the all intermediate step results are returned if `return_all` is True, otherwise only the distribution at `end` is returned.

        Raises
        ------
        TODO: inherit
        """
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

    @d.dedent
    def pull(
        self,
        start: Number,
        end: Number,
        result_key: Optional[str] = None,
        return_all: bool = False,
        scale_by_marginals: bool = True,
        **kwargs: Any,
    ) -> Optional[Union[npt.ArrayLike, Dict[Tuple[Any, Any], npt.ArrayLike]]]:
        """
        Pulls distribution of cells through time

        Parameters
        ----------
        start
            Time point of source distribution (later time point).
        target
            Time point of target distribution (earlier time point).
        result_key
            Key of where to save the result in :class:`anndata.AnnData.obs`. If None the result will be returned.
        return_all
            If True returns all the intermediate masses if pushed through multiple transport plans. If True, the result is returned as a dictionary.

        Returns
        -------
        Depending on `result_key` updates `adata` or returns the result. In the latter case the all intermediate step results are returned if `return_all` is True, otherwise only the distribution at `start` is returned.

        Raises
        ------
        TODO: inherit
        """
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
        """
        Growth rates of the cells
        """
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

    @property
    def proliferation_key(self) -> Optional[str]:
        """
        Key in :attr:`anndata.AnnData.obs` where prior estimate of cell proliferation is saved (created by :meth:`moscot.problems.TemporalProblem.score_genes_for_marginals()`)
        """
        return self._proliferation_key

    @property
    def apoptosis_key(self) -> Optional[str]:
        """
        Key in :attr:`anndata.AnnData.obs` where prior estimate of cell apoptosis is saved (created by :meth:`moscot.problems.TemporalProblem.score_genes_for_marginals()`)
        """
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
        """
        Returns the cost of a cell (see online methods) obtained by the potentials of the optimal transport solution

        Raises
        ------
        NotImplementedError
            If the solver from :class:`moscot.solvers` does not use potentials
        """
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
        """
        Returns the cost of a cell (see online methods) obtained by the potentials of the optimal transport solution

        Raises
        ------
        NotImplementedError
            If the solver from :class:`moscot.solvers` does not use potentials
        """
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


class LineageProblem(TemporalProblem):
    """
    Estimator for modelling time series single cell data based on moslin

    Class handling the computation and downstream analysis of temporal single cell data with lineage prior.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` instance containing the single cell data and corresponding metadata
    solver
        :class:`moscot.solver` instance used to solve the optimal transport problem. Currently, :class:`moscot.solvers.SinkhornSolver` can be used to solve this problem.

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it
    """

    def prepare(
        self,
        time_key: str,
        lineage_attr: Mapping[str, Any] = MappingProxyType({}),
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "LineageProblem":
        """
        Prepares the LineageProblem for it being ready to be solved

        This method executes multiple steps to prepare the problem for the Optimal Transport solver to be ready to solve it

        Parameters
        ----------
        time_key
            Key in :attr:`anndata.AnnData.obs` which defines the time point each cell belongs to. It is supposed to be of numerical data type.
        lineage_attr
            Specifies the way the lineage information is processed. TODO: Specify.
        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps. If None, the data is read from :attr:`anndata.AnnData.X` and for each time point the corresponding PCA space is computed. If `joint_attr` is a string the data is assumed to be found in :attr:`anndata.AnnData.obsm`. If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of :attr:`anndata.AnnData` as a key and the corresponding attribute as a value.

        Returns
        -------
        Self

        Raises
        ------
        KeyError
            If `time_key` is not in :attr:`anndata.AnnData.obs`
        KeyError
            If `joint_attr` is a string and cannot be found in :attr:`anndata.AnnData.obsm`
        """

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

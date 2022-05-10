from abc import ABC
from typing import Any, Type, Union, Literal, Optional, Sequence
from numbers import Number
import logging

import numpy as np
import numpy.typing as npt

from anndata import AnnData
import scanpy as sc

from moscot._docs import d
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._utils import beta, delta, MarkerGenes
from moscot.problems._compound_problem import B
from moscot.problems._multimarginal_problem import MultiMarginalProblem


class MultiMarginalMixin(ABC):
    pass


class BirthDeathMixin(MultiMarginalMixin):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._proliferation_key: Optional[str] = None
        self._apoptosis_key: Optional[str] = None

    @property
    def _base_problem_type(self) -> Type[B]:
        return BirthDeathBaseProblem

    @d.dedent
    def score_genes_for_marginals(
        self,
        gene_set_proliferation: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        gene_set_apoptosis: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        proliferation_key: str = "proliferation",
        apoptosis_key: str = "apoptosis",
        **kwargs: Any,
    ) -> None:
        """
        Compute gene scores to obtain prior knowledge about proliferation and apoptosis.

        This method computes gene scores using :func:`scanpy.tl.score_genes`. Therefore, a list of genes corresponding
        to proliferation and/or apoptosis must be passed.

        Alternatively, proliferation and apoptosis genes for humans and mice are saved in :mod:`moscot`.
        The gene scores will be used in :meth:`moscot.problems.TemporalProblem.prepare` to estimate the initial
        growth rates as suggested in :cite:`schiebinger:19`

        Parameters
        ----------
        gene_set_proliferation
            Set of marker genes for proliferation used in the birth-death process. If marker genes from :mod:`moscot`
            are to be used the corresponding organism must be passed.
        gene_set_apoptosis
            Set of marker genes for apoptosis used in the birth-death process. If marker genes from :mod:`moscot` are
            to be used the corresponding organism must be passed.
        proliferation_key
            Key in :attr:`anndata.AnnData.obs` where to add the genes scores.
        kwargs
            Keyword arguments for :func:`scanpy.tl.score_genes`.

        Returns
        -------
        Returns :class:`moscot.problems.time.TemporalProblem` and updates the following attributes

            - :attr:`proliferation_key`
            - :attr:`apoptosis_key`

        Notes
        -----
        The marker genes in :mod:`moscot` are taken from the following sources:

            - human, proliferation - :cite:`tirosh:16:science`.
            - human, apoptosis - `Hallmark Apoptosis,
              MSigDB <https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_APOPTOSIS>`_.
            - mouse, proliferation - :cite:`tirosh:16:nature`.
            - mouse, apoptosis - `Hallmark P53 Pathway, MSigDB
              <https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_P53_PATHWAY>`_.
        """
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

    @property
    def proliferation_key(self) -> Optional[str]:
        """Key in :attr:`anndata.AnnData.obs` where prior estimate of cell proliferation is saved."""
        return self._proliferation_key

    @property
    def apoptosis_key(self) -> Optional[str]:
        """Key in :attr:`anndata.AnnData.obs` where prior estimate of cell apoptosis is saved."""
        return self._apoptosis_key

    # TODO(michalk8): remove docs in setters
    @proliferation_key.setter
    def proliferation_key(self, value: Optional[str] = None) -> None:
        # TODO(michalk8): add check if present in .obs (if not None)
        self._proliferation_key = value

    @apoptosis_key.setter
    def apoptosis_key(self, value: Optional[str] = None) -> None:
        # TODO(michalk8): add check if present in .obs (if not None)
        self._apoptosis_key = value


@d.dedent
class BirthDeathBaseProblem(MultiMarginalProblem):
    """
    Class handling an optimal transport problem which allows to estimate the marginals with a birth-death process.

    Parameters
    ----------
    %(adata_x)s
    %(adata_y)s
    %(source)s
    %(target)s

    Raises
    ------
    %(MultiMarginalProblem.raises)s
    """

    def __init__(
        self,
        adata_x: AnnData,
        adata_y: AnnData,
        *,
        source: Number,
        target: Number,
        **kwargs: Any,
    ):
        if source >= target:
            raise ValueError(f"{source} is expected to be strictly smaller than {target}.")
        super().__init__(adata_x, adata_y=adata_y, source=source, target=target, **kwargs)

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

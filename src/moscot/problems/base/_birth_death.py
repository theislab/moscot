from typing import Any, Union, Literal, Callable, Optional, Protocol, Sequence, TYPE_CHECKING

import numpy as np

from anndata import AnnData
import scanpy as sc

from moscot._types import ArrayLike
from moscot._logging import logger
from moscot._docs._docs import d
from moscot.problems.time._utils import beta, delta, MarkerGenes
from moscot.problems.base._base_problem import OTProblem

__all__ = ["BirthDeathProblem", "BirthDeathMixin"]


class BirthDeathProtocol(Protocol):
    adata: AnnData
    proliferation_key: Optional[str]
    apoptosis_key: Optional[str]
    _proliferation_key: Optional[str] = None
    _apoptosis_key: Optional[str] = None
    _scaling: Optional[float] = None
    _prior_growth: Optional[ArrayLike] = None

    def score_genes_for_marginals(
        self: "BirthDeathProtocol",
        gene_set_proliferation: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        gene_set_apoptosis: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        proliferation_key: str = "proliferation",
        apoptosis_key: str = "apoptosis",
        **kwargs: Any,
    ) -> "BirthDeathProtocol":
        ...


class BirthDeathProblemProtocol(BirthDeathProtocol, Protocol):
    delta: float
    adata_tgt: AnnData
    a: Optional[ArrayLike]
    b: Optional[ArrayLike]


class BirthDeathMixin:
    """Mixin class for biological problems based on :class:`moscot.problems.mixins.BirthDeathProblem`."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._proliferation_key: Optional[str] = None
        self._apoptosis_key: Optional[str] = None
        self._scaling: Optional[float] = None
        self._prior_growth: Optional[ArrayLike] = None

    @d.dedent
    def score_genes_for_marginals(
        self: BirthDeathProtocol,
        gene_set_proliferation: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        gene_set_apoptosis: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        proliferation_key: str = "proliferation",
        apoptosis_key: str = "apoptosis",
        **kwargs: Any,
    ) -> "BirthDeathProtocol":
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
                    MarkerGenes.proliferation_markers(gene_set_proliferation),  # type: ignore[arg-type]
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
                    MarkerGenes.apoptosis_markers(gene_set_apoptosis),  # type: ignore[arg-type]
                    score_name=apoptosis_key,
                    **kwargs,
                )
            else:
                sc.tl.score_genes(self.adata, gene_set_apoptosis, score_name=apoptosis_key, **kwargs)
            self.apoptosis_key = apoptosis_key
        if gene_set_proliferation is None and gene_set_apoptosis is None:
            logger.info(
                "At least one of `gene_set_proliferation` or `gene_set_apoptosis` must be provided to score genes."
            )

        return self

    @property
    def proliferation_key(self) -> Optional[str]:
        """Key in :attr:`anndata.AnnData.obs` where cell proliferation is stored."""
        return self._proliferation_key

    @proliferation_key.setter
    def proliferation_key(self: BirthDeathProtocol, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find proliferation data in `adata.obs[{key!r}]`.")
        self._proliferation_key = key

    @property
    def apoptosis_key(self) -> Optional[str]:
        """Key in :attr:`anndata.AnnData.obs` where cell apoptosis is stored."""
        return self._apoptosis_key

    @apoptosis_key.setter
    def apoptosis_key(self: BirthDeathProtocol, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find apoptosis data in `adata.obs[{key!r}]`.")
        self._apoptosis_key = key


@d.dedent
class BirthDeathProblem(BirthDeathMixin, OTProblem):
    """
    Class handling an optimal transport problem which allows to estimate the marginals with a birth-death process.

    Parameters
    ----------
    %(adata_x)s
    """

    def _estimate_marginals(
        self: BirthDeathProblemProtocol,
        adata: AnnData,
        source: bool,
        proliferation_key: Optional[str] = None,
        apoptosis_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ArrayLike:
        def estimate(key: Optional[str], *, fn: Callable[..., ArrayLike]) -> ArrayLike:
            if key is None:
                return np.zeros(adata.n_obs, dtype=float)
            try:
                return fn(adata.obs[key].values.astype(float), **kwargs)
            except KeyError:
                raise KeyError(f"Unable to fetch data from `adata.obs[{key}!r]`.") from None

        if proliferation_key is None and apoptosis_key is None:
            raise ValueError("Either `proliferation_key` or `apoptosis_key` must be specified.")
        self.proliferation_key = proliferation_key
        self.apoptosis_key = apoptosis_key

        birth = estimate(proliferation_key, fn=beta)
        death = estimate(apoptosis_key, fn=delta)
        prior_growth = np.exp((birth - death) * self.delta)
        scaling = np.sum(prior_growth)
        normalized_growth = prior_growth / scaling
        if source:
            self._scaling = scaling
            self._prior_growth = prior_growth

        return normalized_growth if source else np.full(self.adata_tgt.n_obs, fill_value=np.mean(normalized_growth))

    # TODO(michalk8): temporary fix to satisfy the mixin, consider removing the mixin
    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self.adata_src

    @property
    def prior_growth_rates(self) -> Optional[ArrayLike]:
        """Return the prior estimate of growth rates of the cells in the source distribution."""
        if self._prior_growth is None:
            return None
        return np.power(self._prior_growth, 1.0 / self.delta)

    @property
    def posterior_growth_rates(self) -> Optional[ArrayLike]:
        """Return the posterior estimate of growth rates of the cells in the source distribution."""
        if self.solution.a is None:  # type: ignore[union-attr]
            return None
        if self.delta is None:
            return self.solution.a * self.adata.n_obs
        return np.power(self.solution.a * self._scaling, 1.0 / self.delta)  # type: ignore[union-attr, operator]

    @property
    def delta(self) -> float:
        """TODO."""
        if TYPE_CHECKING:
            assert isinstance(self._src_key, float)
            assert isinstance(self._tgt_key, float)
        return self._tgt_key - self._src_key

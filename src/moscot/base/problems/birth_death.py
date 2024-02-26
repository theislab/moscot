from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
)

import numpy as np

import scanpy as sc
from anndata import AnnData

from moscot._logging import logger
from moscot._types import ArrayLike
from moscot.base.problems.problem import OTProblem
from moscot.utils.data import apoptosis_markers, proliferation_markers

__all__ = ["BirthDeathProblem", "BirthDeathMixin"]


class BirthDeathProtocol(Protocol):  # noqa: D101
    adata: AnnData
    proliferation_key: Optional[str]
    apoptosis_key: Optional[str]
    _proliferation_key: Optional[str]
    _apoptosis_key: Optional[str]
    _scaling: float
    _prior_growth: Optional[ArrayLike]

    def score_genes_for_marginals(  # noqa: D102
        self: "BirthDeathProtocol",
        gene_set_proliferation: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        gene_set_apoptosis: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        proliferation_key: str = "proliferation",
        apoptosis_key: str = "apoptosis",
        **kwargs: Any,
    ) -> "BirthDeathProtocol":
        ...


class BirthDeathProblemProtocol(BirthDeathProtocol, Protocol):  # noqa: D101
    delta: float
    adata_tgt: AnnData
    a: Optional[ArrayLike]
    b: Optional[ArrayLike]


class BirthDeathMixin:
    """Mixin class used to estimate cell proliferation and apoptosis.

    Parameters
    ----------
    args
        Positional arguments.
    kwargs
        Keyword arguments.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._proliferation_key: Optional[str] = None
        self._apoptosis_key: Optional[str] = None
        self._scaling: float = 1.0
        self._prior_growth: Optional[ArrayLike] = None

    def score_genes_for_marginals(
        self,  # type: BirthDeathProblemProtocol
        gene_set_proliferation: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        gene_set_apoptosis: Optional[Union[Literal["human", "mouse"], Sequence[str]]] = None,
        proliferation_key: str = "proliferation",
        apoptosis_key: str = "apoptosis",
        **kwargs: Any,
    ) -> "BirthDeathMixin":
        """Compute gene scores to obtain prior knowledge about proliferation and apoptosis.

        The gene scores can be used in :meth:`~moscot.base.problems.BirthDeathProblem.estimate_marginals`
        to estimate the initial growth rates as suggested in :cite:`schiebinger:19`

        Parameters
        ----------
        gene_set_proliferation
            Set of proliferation marker genes. If a :class:`str`, it should
            correspond to the organism in :func:`~moscot.utils.data.proliferation_markers`.
        gene_set_apoptosis
            Set of apoptosis marker genes. If a :class:`str`, it should
            correspond to the organism in :func:`~moscot.utils.data.apoptosis_markers`.
        proliferation_key
            Key in :attr:`~anndata.AnnData.obs` where to store the proliferation scores.
        apoptosis_key
            Key in :attr:`~anndata.AnnData.obs` where to store the apoptosis scores.
        kwargs
            Keyword arguments for :func:`~scanpy.tl.score_genes`.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`proliferation_key` - key in :attr:`~anndata.AnnData.obs` where proliferation scores are stored.
        - :attr:`apoptosis_key` - key in :attr:`~anndata.AnnData.obs` where apoptosis scores are stored.
        """
        if isinstance(gene_set_proliferation, str):
            gene_set_proliferation = proliferation_markers(gene_set_proliferation)  # type: ignore[arg-type]
        if gene_set_proliferation is not None:
            sc.tl.score_genes(self.adata, gene_set_proliferation, score_name=proliferation_key, **kwargs)
            self.proliferation_key = proliferation_key
        else:
            self.proliferation_key = None

        if isinstance(gene_set_apoptosis, str):
            gene_set_apoptosis = apoptosis_markers(gene_set_apoptosis)  # type: ignore[arg-type]
        if gene_set_apoptosis is not None:
            sc.tl.score_genes(self.adata, gene_set_apoptosis, score_name=apoptosis_key, **kwargs)
            self.apoptosis_key = apoptosis_key
        else:
            self.apoptosis_key = None

        if self.proliferation_key is None and self.apoptosis_key is None:
            logger.warning(
                "At least one of `gene_set_proliferation` or `gene_set_apoptosis` must be provided to score genes."
            )

        return self  # type: ignore[return-value]

    @property
    def proliferation_key(self) -> Optional[str]:
        """Key in :attr:`~anndata.AnnData.obs` where cell proliferation is stored."""
        return self._proliferation_key

    @proliferation_key.setter
    def proliferation_key(self: BirthDeathProtocol, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find proliferation data in `adata.obs[{key!r}]`.")
        self._proliferation_key = key

    @property
    def apoptosis_key(self) -> Optional[str]:
        """Key in :attr:`~anndata.AnnData.obs` where cell apoptosis is stored."""
        return self._apoptosis_key

    @apoptosis_key.setter
    def apoptosis_key(self: BirthDeathProtocol, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find apoptosis data in `adata.obs[{key!r}]`.")
        self._apoptosis_key = key


class BirthDeathProblem(BirthDeathMixin, OTProblem):
    """:term:`OT` problem used to estimate the :term:`marginals` with the
    `birth-death process <https://en.wikipedia.org/wiki/Birth%E2%80%93death_process>`_.

    Parameters
    ----------
    args
        Positional arguments for :class:`~moscot.base.problems.OTProblem`.
    kwargs
        Keyword arguments for :class:`~moscot.base.problems.OTProblem`.
    """  # noqa: D205

    def estimate_marginals(
        self,  # type: BirthDeathProblemProtocol
        adata: AnnData,
        source: bool,
        proliferation_key: Optional[str] = None,
        apoptosis_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Estimate the source or target :term:`marginals` based on marker genes, either with the
        `birth-death process <https://en.wikipedia.org/wiki/Birth%E2%80%93death_process>`_,
        as suggested in :cite:`schiebinger:19`, or with an exponential kernel.

        See :meth:`score_genes_for_marginals` on how to compute the proliferation and apoptosis scores.

        Parameters
        ----------
        adata
            Annotated data object.
        source
            Whether to estimate the source or the target :term:`marginals`.
        proliferation_key
            Key in :attr:`~anndata.AnnData.obs` where proliferation scores are stored.
        apoptosis_key
            Key in :attr:`~anndata.AnnData.obs` where apoptosis scores are stored.
        kwargs
            Keyword arguments for :func:`~moscot.base.problems.birth_death.beta` and
            :func:`~moscot.base.problems.birth_death.delta`.

        Returns
        -------
        The estimated source or target marginals of shape ``[n,]`` or ``[m,]``, depending on the ``source``.
        If ``source = True``, also updates the following fields:

        - :attr:`prior_growth_rates` - prior estimate of the source growth rates.

        Examples
        --------
        - See :doc:`../notebooks/examples/problems/800_score_genes_for_marginals`
            on examples how to use :meth:`~moscot.problems.time.TemporalProblem.score_genes_for_marginals`.

        """  # noqa: D205

        def estimate(key: Optional[str], *, fn: Callable[..., ArrayLike], **kwargs: Any) -> ArrayLike:
            if key is None:
                return np.zeros(adata.n_obs, dtype=float)
            try:
                return fn(adata.obs[key].values.astype(float), **kwargs)
            except KeyError:
                raise KeyError(f"Unable to get data from `adata.obs[{key}!r]`.") from None

        if proliferation_key is None and apoptosis_key is None:
            raise ValueError("At least one of `proliferation_key` or `apoptosis_key` must be specified.")

        # TODO(michalk8): why does this need to be set?
        self.proliferation_key = proliferation_key
        self.apoptosis_key = apoptosis_key

        if "scaling" in kwargs:
            beta_fn = delta_fn = lambda x, *_, **__: x
            scaling = kwargs["scaling"]
        else:
            beta_fn, delta_fn = beta, delta
            scaling = 1.0
        birth = estimate(proliferation_key, fn=beta_fn, **kwargs)
        death = estimate(apoptosis_key, fn=delta_fn, **kwargs)

        prior_growth = np.exp((birth - death) * self.delta / scaling)

        scaling = np.sum(prior_growth)
        normalized_growth = prior_growth / scaling
        if source:
            self._scaling = scaling
            self._prior_growth = prior_growth
            return normalized_growth

        return np.full(self.adata_tgt.n_obs, fill_value=np.mean(normalized_growth))

    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self.adata_src

    @property
    def prior_growth_rates(self) -> Optional[ArrayLike]:
        """Prior estimate of the source growth rates."""
        if self._prior_growth is None:
            return None
        return np.asarray(np.power(self._prior_growth, 1.0 / self.delta))

    @property
    def posterior_growth_rates(self) -> Optional[ArrayLike]:
        """Posterior estimate of the source growth rates."""
        if self.solution is None:
            return None
        if self.delta is None:
            return self.solution.a * self.adata.n_obs
        return np.asarray(self.solution.a * self._scaling) ** (1.0 / self.delta)

    @property
    def delta(self) -> float:
        """Time difference between the source and the target."""
        if TYPE_CHECKING:
            assert isinstance(self._src_key, float)
            assert isinstance(self._tgt_key, float)
        return self._tgt_key - self._src_key


def _logistic(x: ArrayLike, L: float, k: float, center: float = 0) -> ArrayLike:
    """Logistic function."""
    return L / (1 + np.exp(-k * (x - center)))


def _gen_logistic(p: ArrayLike, sup: float, inf: float, center: float, width: float) -> ArrayLike:
    """Shifted logistic function."""
    return inf + _logistic(p, L=sup - inf, k=4.0 / width, center=center)


def beta(
    p: ArrayLike,
    beta_max: float = 1.7,
    beta_min: float = 0.3,
    beta_center: float = 0.25,
    beta_width: float = 0.5,
    **_: Any,
) -> ArrayLike:
    """Birth process."""
    return _gen_logistic(p, beta_max, beta_min, beta_center, beta_width)


def delta(
    a: ArrayLike,
    delta_max: float = 1.7,
    delta_min: float = 0.3,
    delta_center: float = 0.1,
    delta_width: float = 0.2,
    **_: Any,
) -> ArrayLike:
    """Death process."""
    return _gen_logistic(a, delta_max, delta_min, delta_center, delta_width)

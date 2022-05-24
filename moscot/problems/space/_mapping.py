from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Mapping, Optional, Sequence

from typing_extensions import Literal

import numpy.typing as npt

from anndata import AnnData

from moscot._docs import d
from moscot.analysis_mixins import SpatialMappingAnalysisMixin
from moscot.problems._base_problem import OTProblem
from moscot.problems._subset_policy import Axis_t, DummyPolicy, ExternalStarPolicy
from moscot.problems._compound_problem import B, SingleCompoundProblem


@d.dedent
class MappingProblem(SingleCompoundProblem, SpatialMappingAnalysisMixin):
    """
    Class for mapping single cell omics data onto spatial data, based on :cite:`nitzan2019`.

    The `MappingProblem` allows to match single cell and spatial omics data via optimal transport.

    Parameters
    ----------
    adata_sc
        Instance of :class:`anndata.AnnData` containing the single cell data.
    adata_sp
        Instance of :class:`anndata.AnnData` containing the spatial data.

    Examples
    --------
    See notebook TODO(@giovp) LINK NOTEBOOK for how to use it
    """

    def __init__(self, adata_sc: AnnData, adata_sp: AnnData, **kwargs: Any):
        super().__init__(adata_sp, **kwargs)
        self._adata_sc = adata_sc
        # TODO(michalk8): rename to common_vars?
        self.filtered_vars: Optional[Sequence[str]] = None

    def _create_policy(
        self,
        policy: Literal["external_star"] = "external_star",
        key: Optional[str] = None,
        axis: Axis_t = "obs",
        **kwargs: Any,
    ) -> Union[DummyPolicy, ExternalStarPolicy]:
        """Private class to create DummyPolicy if no batches are present n the spatial anndata."""
        if key is None:
            return DummyPolicy(self.adata, axis=axis, **kwargs)
        return ExternalStarPolicy(self.adata, key=key, axis=axis, **kwargs)

    def _create_problem(
        self,
        src: Any,
        tgt: Any,
        src_mask: npt.ArrayLike,
        tgt_mask: npt.ArrayLike,
        **kwargs: Any,
    ) -> B:
        """Private class to mask anndatas."""
        adata_sp = self._mask(src_mask)
        return self._base_problem_type(
            adata_sp[:, self.filtered_vars] if self.filtered_vars is not None else adata_sp,
            self.adata_sc[:, self.filtered_vars] if self.filtered_vars is not None else self.adata_sc,
            **kwargs,
        )

    @d.dedent
    def prepare(
        self,
        sc_attr: Union[str, Mapping[str, Any]],
        batch_key: Optional[str] = None,
        spatial_key: Union[str, Mapping[str, Any]] = "spatial",
        var_names: Optional[Sequence[Any]] = None,
        joint_attr: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        **kwargs: Any,
    ) -> "MappingProblem":
        """
        Prepare the :class:`moscot.problems.space.MappingProblem`.

        This method prepares the data to be passed to the optimal transport solver.

        Parameters
        ----------
        sc_attr
            Specifies the attributes of the single cell adata.

        %(batch_key)s
        %(spatial_key)s

        var_names
            List of shared features to be used for the linear problem. If None, it defaults to the intersection
            between ``adata_sc`` and ``adata_sp``. If an empty list is pass, it defines a quadratic problem.
        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps.
            If None, ``var_names`` is not an empty list, and ``adata_sc`` and ``adata_sp``
            share some genes in common, the corresponding PCA space is computed.
            If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of
            :attr:`anndata.AnnData` as a key and the corresponding attribute as a value.

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`
        """
        x = {"attr": "obsm", "key": spatial_key} if isinstance(spatial_key, str) else spatial_key
        y = {"attr": "obsm", "key": sc_attr} if isinstance(sc_attr, str) else sc_attr

        self.filtered_vars = var_names
        if self.filtered_vars is not None:
            if joint_attr is not None:
                kwargs["xy"] = joint_attr
            else:
                kwargs["callback"] = "local-pca"
                kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}

        return super().prepare(x=x, y=y, policy="external_star", key=batch_key, **kwargs)

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-3,
        scale_cost: str = "mean",
        **kwargs: Any,
    ) -> "MappingProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.MappingProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(scale_cost)s

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`
        """
        return super().solve(alpha=alpha, epsilon=epsilon, scale_cost=scale_cost, **kwargs)

    @property
    def adata_sc(self) -> AnnData:
        """Return single cell adata."""
        return self._adata_sc

    @property
    def adata_sp(self) -> AnnData:
        """Return spatial adata. Alias for :attr:`adata`."""
        return self.adata

    @property
    def filtered_vars(self) -> Optional[Sequence[Any]]:
        """Return filtered variables."""
        return self._filtered_vars

    @filtered_vars.setter
    def filtered_vars(self, value: Optional[Sequence[str]]) -> None:
        """Return filtered variables."""
        self._filtered_vars = self._filter_vars(var_names=value)

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return ("external_star",)

from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Mapping, Optional, Sequence

from typing_extensions import Literal

import numpy.typing as npt

from anndata import AnnData

from moscot.problems._base_problem import OTProblem
from moscot.problems._subset_policy import Axis_t, DummyPolicy, ExternalStarPolicy
from moscot.mixins._spatial_analysis import SpatialMappingAnalysisMixin
from moscot.problems._compound_problem import B, SingleCompoundProblem


class MappingProblem(SingleCompoundProblem, SpatialMappingAnalysisMixin):
    """Mapping problem."""

    def __init__(self, adata_sc: AnnData, adata_sp: AnnData, **kwargs: Any):
        """Init method."""
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
        adata_sp = self._mask(src_mask)
        return self._base_problem_type(
            adata_sp[:, self.filtered_vars] if self.filtered_vars is not None else adata_sp,
            self.adata_sc[:, self.filtered_vars] if self.filtered_vars is not None else self.adata_sc,
            **kwargs,
        )

    def prepare(
        self,
        sc_attr: Union[str, Mapping[str, Any]],
        spatial_key: str = "spatial",
        joint_attr: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        var_names: Optional[Sequence[Any]] = None,
        batch_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "MappingProblem":
        """Prepare method."""
        x = {"attr": "obsm", "key": spatial_key}
        y = {"attr": "obsm", "key": sc_attr} if isinstance(sc_attr, str) else sc_attr

        self.filtered_vars = var_names
        if self.filtered_vars is not None:
            if joint_attr is not None:
                kwargs["xy"] = joint_attr
            else:
                kwargs["callback"] = "local-pca"
                kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}

        return super().prepare(x=x, y=y, policy="external_star", key=batch_key, **kwargs)

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

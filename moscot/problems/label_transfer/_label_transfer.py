from types import MappingProxyType
from typing import Any, Type, Union, Literal, Mapping, Optional, Sequence

from anndata import AnnData

from moscot._types import ArrayLike
from moscot._constants._constants import Policy
from moscot.problems._subset_policy import Axis_t, DummyPolicy, SequentialPolicy
from moscot.problems.base._base_problem import OTProblem
from moscot.problems.base._compound_problem import B, K, CompoundProblem
from moscot.problems.label_transfer._mixins import LabelMixin

__all__ = "LabelProblem"


class LabelProblem(LabelMixin[K, OTProblem], CompoundProblem[K, OTProblem]):
    def __init__(self, adata_unlabelled: AnnData, adata_labelled: AnnData, **kwargs: Any):
        super().__init__(adata_labelled, **kwargs)
        self._adata_unlabelled = adata_unlabelled
        # TODO(michalk8): rename to common_vars?
        self._filtered_vars: Optional[Sequence[str]] = None

    def _create_problem(
        self,
        src_mask: ArrayLike,
        tgt_mask: ArrayLike,
        **kwargs: Any,
    ) -> B:
        """Private class to mask anndatas."""
        adata_labelled = self._mask(src_mask)
        return self._base_problem_type(  # type: ignore[return-value]
            adata_labelled[:, self.filtered_vars] if self.filtered_vars is not None else adata_labelled,
            self.adata_unlabelled[:, self.filtered_vars] if self.filtered_vars is not None else self.adata_unlabelled,
            **kwargs,
        )

    def _create_policy(  # type: ignore[override]
        self,
        policy: Literal[Policy.SEQUENTIAL] = Policy.SEQUENTIAL,
        key: Optional[str] = None,
        axis: Axis_t = "obs",
        **kwargs: Any,
    ) -> Union[DummyPolicy, SequentialPolicy[K]]:
        """Private class to create DummyPolicy if no batches are present n the spatial anndata."""
        if key is None:
            return DummyPolicy(self.adata, axis=axis, **kwargs)
        return SequentialPolicy(self.adata, key=key, axis=axis, **kwargs)

    def prepare(  # type: ignore[override]
        self,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        GW_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        filtered_vars: Optional[Sequence] = None,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "LabelProblem[K]":
        self.filtered_vars = (
            filtered_vars
            if filtered_vars is not None
            else list(set(self.adata_labelled.var.index).intersection(self.adata_unlabelled.var.index))
        )
        if joint_attr is None:
            if "callback" not in kwargs:
                kwargs["callback"] = "local-pca"
            else:
                kwargs["callback"] = kwargs["callback"]
            kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}
        elif isinstance(joint_attr, str):
            kwargs["xy"] = {
                "x_attr": "obsm",
                "x_key": joint_attr,
                "y_attr": "obsm",
                "y_key": joint_attr,
            }
        elif isinstance(joint_attr, Mapping):
            kwargs["xy"] = joint_attr
        else:
            raise TypeError("TODO")

        if GW_attr is not None:
            if isinstance(joint_attr, str):
                gw_attr = {
                    "attr": "obsm",
                    "key": GW_attr,
                }
            elif isinstance(GW_attr, Mapping):
                gw_attr = GW_attr
            else:
                raise TypeError("TODO")
            kwargs["x"] = kwargs["y"] = gw_attr

        return super().prepare(
            key=None,
            policy="sequential",
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> str:
        return "sequential"

    @property
    def adata_labelled(self) -> AnnData:
        return self.adata

    @property
    def adata_unlabelled(self) -> AnnData:
        return self._adata_unlabelled

    @property
    def filtered_vars(self) -> Optional[Sequence[str]]:
        """Return filtered variables."""
        return self._filtered_vars

    @filtered_vars.setter
    def filtered_vars(self, value: Optional[Sequence[str]]) -> None:
        """Return filtered variables."""
        self._filtered_vars = value

    @property
    def _other_adata(self) -> Optional[AnnData]:
        return self._adata_unlabelled

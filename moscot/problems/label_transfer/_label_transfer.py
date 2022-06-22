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
            adata_labelled[self.adata_labelled.obs[self.key_labelled.isin(self.subset_labelled)], self.filtered_vars] if self.filtered_vars is not None else adata_labelled,
            self.adata_unlabelled[self.key_unlabelled.isin(self.subset_unlabelled), self.filtered_vars] if self.filtered_vars is not None else self.adata_unlabelled,
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
        key_labelled: Union[str, Mapping[str, Sequence[Any]]] = "labelled",
        key_unlabelled: Union[str, Mapping[str, Sequence[Any]]] = "unlabelled",
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        GW_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        filtered_vars: Optional[Sequence] = None,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "LabelProblem[K]":
        self._handle_keys(key_labelled, key_unlabelled)

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

    def _handle_keys(self, key_labelled: Union[str, Mapping[str, Sequence[Any]]] = "labelled", key_unlabelled: Union[str, Mapping[str, Sequence[Any]]] = "labelled") -> None:
        if isinstance(key_labelled, str):
            if key_labelled not in self.adata.obs.columns:
                self.adata.obs[key_labelled] = 0
                self.key_labelled = key_labelled
                self.subset_labelled = (0,)
        elif isinstance(key_labelled, dict):
            if len(key_labelled) == 0:
                raise ValueError("TODO: EMPTY dict")
            elif len(key_labelled) > 1:
                raise ValueError("TODO: dict must be of length 1")
            _key_labelled = key_labelled.keys()[0]
            if _key_labelled not in self.adata.obs.columns:
                raise KeyError("TODO: key not in adata.obs.")
            self.key_labelled = _key_labelled
            self.subset_labelled = list(key_labelled.values())

        if isinstance(key_unlabelled, str):
            if key_unlabelled not in self.adata_unlabelled.obs.columns:
                self.adata_unlabelled.obs[key_unlabelled] = 0
                self.key_unlabelled = key_unlabelled
                self.subset_unlabelled = (1,)
        elif isinstance(key_unlabelled, dict):
            if len(key_unlabelled) == 0:
                raise ValueError("TODO: EMPTY dict")
            elif len(key_unlabelled) > 1:
                raise ValueError("TODO: dict must be of length 1")
            _key_unlabelled = key_unlabelled.keys()[0]
            if _key_unlabelled not in self.adata_unlabelled.obs.columns:
                raise KeyError("TODO: key not in adata.obs.")
            self.key_unlabelled = _key_unlabelled
            self.subset_unlabelled = list(key_unlabelled.values())

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

from types import MappingProxyType
from typing import Any, Type, Union, Mapping, Optional

from moscot._types import Numeric_t
from moscot.problems.base._birth_death import BirthDeathProblem
from moscot.problems.base._base_problem import OTProblem
from moscot.problems.base._compound_problem import B, K, CompoundProblem
from moscot.problems.label_transfer._mixins import LabelMixin


class LabelProblem(LabelMixin[K, OTProblem], CompoundProblem[Numeric_t, BirthDeathProblem]):
    def prepare(
        self,
        batch_key: str,
        labelled_batch: Any,  # TODO: specify
        batch_to_label: Any,  # TODO: specify
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        GW_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "LabelProblem":
        if self.adata.obs[batch_key].nunique() != 2:
            raise ValueError("TODO: Samples must correspond exactly to 2 batches")
        if labelled_batch not in self.adata.obs[batch_key].cat.categories:
            raise ValueError(f"TODO: {labelled_batch} not in `adata.obs`")
        if batch_to_label not in self.adata.obs[batch_key].cat.categories:
            raise ValueError(f"TODO: {batch_to_label} not in `adata.obs`")
        self.labelled_batch = labelled_batch
        self.batch_to_label = batch_to_label

        self.batch_key = batch_key
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
            key=batch_key,
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

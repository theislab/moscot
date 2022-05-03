from types import MappingProxyType
from typing import Any, Type, Tuple, Mapping, Optional

from typing_extensions import Literal

from anndata import AnnData

from moscot.problems._base_problem import OTProblem
from moscot.mixins._spatial_analysis import SpatialAlignmentAnalysisMixin
from moscot.problems._compound_problem import B, SingleCompoundProblem


class AlignmentProblem(SingleCompoundProblem, SpatialAlignmentAnalysisMixin):
    """Spatial alignment problem."""

    def __init__(self, adata: AnnData):
        """Init method."""
        super().__init__(adata)
        self._spatial_key: Optional[str] = None

    def prepare(
        self,
        batch_key: str,
        spatial_key: str = "spatial",
        joint_attr: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        policy: Literal["sequential", "star"] = "sequential",
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> "AlignmentProblem":
        """Prepare method."""
        self._spatial_key = spatial_key
        # TODO(michalk8): check for spatial key
        x = y = {"attr": "obsm", "key": self.spatial_key, "tag": "point_cloud"}

        if joint_attr is None:
            kwargs["callback"] = "local-pca"
            kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}

        return super().prepare(x=x, y=y, xy=joint_attr, policy=policy, key=batch_key, reference=reference, **kwargs)

    @property
    def spatial_key(self) -> Optional[str]:
        """Return problems."""
        return self._spatial_key

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "star"

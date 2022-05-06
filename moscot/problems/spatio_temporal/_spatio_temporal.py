from types import MappingProxyType
from typing import Any, Tuple, Mapping, Optional

from typing_extensions import Literal

from moscot.problems.time._lineage import BirthDeathMixin
from moscot.problems.space._alignment import AlignmentProblem
from moscot.analysis_mixins._time_analysis import TemporalAnalysisMixin
from moscot.analysis_mixins._spatial_analysis import SpatialAlignmentAnalysisMixin


class SpatioTemporalProblem(BirthDeathMixin, AlignmentProblem, SpatialAlignmentAnalysisMixin, TemporalAnalysisMixin):
    """Spatio-Temporal problem."""

    def prepare(
        self,
        time_key: str,
        spatial_key: str = "spatial",
        joint_attr: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> "AlignmentProblem":
        """Prepare method."""
        self._SPATIAL_KEY = spatial_key
        self._TEMPORAL_KEY = time_key
        # TODO(michalk8): check for spatial key
        x = y = {"attr": "obsm", "key": self.spatial_key, "tag": "point_cloud"}

        if joint_attr is None:
            kwargs["callback"] = "local-pca"
            kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}

        return super().prepare(x=x, y=y, xy=joint_attr, policy=policy, key=time_key, reference=reference, **kwargs)

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "triu", "tril", "explicit"

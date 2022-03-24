from abc import ABC
from typing import Any

import numpy.typing as npt


# TODO(michalk8): need to think about this a bit more
class AnalysisMixin(ABC):
    """Analysis Mixin class."""

    def _interpolate_transport(
        self, start: Any, end: Any, forward: bool = True, normalize: bool = True
    ) -> npt.ArrayLike:
        """Interpolate transport matrix."""
        steps = self._policy.plan(start=start, end=end)[start, end]
        if len(steps) == 1:
            return self._problems[steps[0]].solution._scale_transport_by_marginals(forward=forward)
        # TODO: find way to push/pull across solutions
        tmap = self._problems[steps[0]].solution._scale_transport_by_marginals(forward=True)
        for i in range(len(steps) - 1):
            tmap = tmap @ self._problems[steps[i + 1]].solution._scale_transport_by_marginals(forward=True)
        if normalize:
            if forward:
                return tmap / tmap.sum(1)[:, None]
            return tmap / tmap.sum(0)[None, :]
        return tmap


# TODO(michalk8): CompoundAnalysisMixin?

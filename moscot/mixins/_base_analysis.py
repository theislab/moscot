from abc import ABC

import numpy as np
import numpy.typing as npt


# TODO(michalk8): need to think about this a bit more
class AnalysisMixin(ABC):
    def _compute_transport_map(self, start: int, end: int, normalize: bool = True) -> npt.ArrayLike:
        if (start, end) not in self._problems.keys():
            steps = self._policy.plan(start=start, end=end)[start, end]
            transition_matrix = self._problems[steps[0]].solution.transport_matrix
            for i in range(len(steps) - 1):
                transition_matrix @= self._problems[steps[i + 1]].solution.transport_matrix
            if normalize:
                transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]
            return transition_matrix
        else:
            transition_matrix = self.solution[
                (start, end)
            ].solution.transport_matrix  # TODO(@MUCDK) report bug, one "solution" too much
            if normalize:
                transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]
            return transition_matrix


# TODO(michalk8): CompoundAnalysisMixin?

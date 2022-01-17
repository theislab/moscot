from moscot.mixins import TemporalAnalysisMixin
from moscot.problems._compound_problem import CompoundProblem


class TemporalProblem(TemporalAnalysisMixin, CompoundProblem):
    # TODO(michalk8): decide how to pass marginals
    # maybe require for BaseProblem as
    # _compute_marginals(self, **kwargs: Any) -> Tuple[Optional[npt.ArrayLike], Optional[npt.ArrayLike]]:
    # as an optional extra step before prepare and in prepare, specify keys
    def estimate_marginals(self):
        pass
